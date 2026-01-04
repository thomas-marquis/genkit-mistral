package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/thomas-marquis/genkit-mistral/mistral"
	mistralclient "github.com/thomas-marquis/mistral-client/mistral"
)

type Recipe struct {
	Title       string
	Ingredients []string
	Steps       []string
}

func (r Recipe) String() string {
	sb := strings.Builder{}
	sb.WriteString("# ")
	sb.WriteString(r.Title)
	sb.WriteString("\n")
	sb.WriteString("\n## Ingredients:\n")
	for _, ing := range r.Ingredients {
		sb.WriteString(ing)
		sb.WriteString("\n")
	}
	sb.WriteString("\n## Steps:\n")
	for _, step := range r.Steps {
		sb.WriteString(step)
		sb.WriteString("\n")
	}
	return sb.String()
}

type GroceryList struct {
	sync.Mutex
	indexedList  map[int]string
	currentIndex int
}

func NewGroceryList() *GroceryList {
	return &GroceryList{
		indexedList:  make(map[int]string),
		currentIndex: 100,
	}
}

func (g *GroceryList) Add(item string) int {
	g.Lock()
	defer g.Unlock()
	g.indexedList[g.currentIndex] = item
	g.currentIndex++
	return g.currentIndex - 1
}

func (g *GroceryList) Update(id int, item string) error {
	g.Lock()
	defer g.Unlock()
	if _, ok := g.indexedList[id]; !ok {
		return fmt.Errorf("item with ID %d not found", id)
	}
	g.indexedList[id] = item
	return nil
}

func (g *GroceryList) Delete(id int) error {
	g.Lock()
	defer g.Unlock()
	if _, ok := g.indexedList[id]; !ok {
		return fmt.Errorf("item with ID %d not found", id)
	}
	delete(g.indexedList, id)
	return nil
}

func (g *GroceryList) Len() int {
	return len(g.indexedList)
}

func (g *GroceryList) String() string {
	sb := strings.Builder{}
	for id, item := range g.indexedList {
		sb.WriteString(fmt.Sprintf("- ID=%d ; Item=%s\n", id, item))
	}
	return sb.String()
}

func main() {
	apiKey := os.Getenv("MISTRAL_API_KEY")
	if apiKey == "" {
		panic("Please set MISTRAL_API_KEY environment variable")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(mistral.NewPlugin(apiKey)))

	genkit.DefinePrompt(g, "recipePrompt",
		ai.WithSystem(`You are a professional and experienced chef. You are creative and pragmatic.`),
		ai.WithOutputType(Recipe{}),
		ai.WithConfig(mistralclient.CompletionConfig{
			Temperature: 0.7,
		}),
		ai.WithModelName("mistral/mistral-small-latest"),
		ai.WithPrompt(`Create a complete recipe according to the following instructions:
{{instructions}}`))

	createRecipe := genkit.DefineFlow(g, "recipeCreator", func(ctx context.Context, in string) (Recipe, error) {
		var recipe Recipe
		res, err := genkit.LookupPrompt(g, "recipePrompt").Execute(ctx,
			ai.WithInput(map[string]interface{}{
				"instructions": in,
			}))
		if err != nil {
			return Recipe{}, err
		}
		if err := res.Output(&recipe); err != nil {
			return Recipe{}, err
		}

		return recipe, nil
	})

	groceryList := NewGroceryList()

	genkit.DefineTool(g, "groceryListGet", "get the current content of the list", func(ctx *ai.ToolContext, input any) (string, error) {
		fmt.Println("groceryListGet")
		if groceryList.Len() == 0 {
			return "The list is empty", nil
		}
		return groceryList.String(), nil
	})

	type groceryListAddInput struct {
		Item string `jsonschema_description:"The item to add to the list and the quantity indication"`
	}

	genkit.DefineTool(g, "groceryListAdd", "add an item to the list", func(ctx *ai.ToolContext, input groceryListAddInput) (string, error) {
		fmt.Printf("groceryListAdd: %s\n", input.Item)
		groceryList.Add(input.Item)
		return "ok", nil
	})

	type groceryListDeleteInput struct {
		ID int `jsonschema_description:"The ID of the item to remove from the list"`
	}

	genkit.DefineTool(g, "groceryListDelete", "remove an item from the grocery list", func(ctx *ai.ToolContext, input groceryListDeleteInput) (string, error) {
		fmt.Printf("groceryListDelete: %d\n", input.ID)
		if err := groceryList.Delete(input.ID); err != nil {
			return "", err
		}
		return "ok", nil
	})

	type groceryListUpdateInput struct {
		ID          int    `jsonschema_description:"The ID of the item to update in the list"`
		UpdatedItem string `jsonschema_description:"The new item to replace the old one with the updated label and/or quantity indication"`
	}

	genkit.DefineTool(g, "groceryListUpdate", "update an item in the grocery list", func(ctx *ai.ToolContext, input groceryListUpdateInput) (string, error) {
		fmt.Printf("groceryListUpdate: %d %s\n", input.ID, input.UpdatedItem)
		if err := groceryList.Update(input.ID, input.UpdatedItem); err != nil {
			return "", err
		}
		return fmt.Sprintf("UpdatedItem %d updated in the list", input.ID), nil
	})

	genkit.DefinePrompt(g, "groceryListPrompt",
		ai.WithSystem(`You are a grocery list manager. Your role is to keep it as clean and coherent as possible. 
Instructions:
- Avoid duplication: if an item already exists in the list, update it instead of adding a new one.
- Get the list content times to times
- Ensure you don't forget any item on the list.
- Use this format for list items: "<label>, <quantity>"
- Avoid recipe-related comment in the list item label, keep it simple`),
		ai.WithPrompt(`Add or update the grocery list with the following items:
{{#each items}}- {{this}}{{/each}}
`),
		ai.WithModelName("mistral/mistral-medium-latest"),
		ai.WithConfig(mistralclient.CompletionConfig{
			Temperature:       0.1,
			ParallelToolCalls: true,
		}),
		ai.WithTools(
			genkit.LookupTool(g, "groceryListGet"),
			genkit.LookupTool(g, "groceryListAdd"),
			genkit.LookupTool(g, "groceryListDelete"),
			genkit.LookupTool(g, "groceryListUpdate"),
		),
	)

	groceryListManagerFlow := genkit.DefineFlow(g, "groceryListManagerFlow", func(ctx context.Context, in []string) (string, error) {
		res, err := genkit.LookupPrompt(g, "groceryListPrompt").Execute(ctx,
			ai.WithInput(map[string]interface{}{
				"items": in,
			}),
			ai.WithReturnToolRequests(false),
			ai.WithMaxTurns(25),
		)
		if err != nil {
			return "", err
		}
		return res.Text(), nil
	})

	genkit.DefinePrompt(g, "menuPrompt",
		ai.WithModelName("mistral/mistral-small-latest"),
		ai.WithConfig(mistralclient.CompletionConfig{
			Temperature: 0.7,
		}),
		ai.WithSystem(`You are a menu planner for an individual. Just give a list of courses without any details according the the constraints given by the user.
Don't create the full recipe, just a short description of the meal. Examples:
- Grilled Salmon with Quinoa and Steamed Broccoli
- Chickpea and Spinach Curry with Brown Rice
- Stuffed Bell Peppers with Lean Turkey and Quinoa

Respect the number of meal to create given by the user.
'
`),
		ai.WithPrompt(`Create a {{nb}}-meals menu that respect this constraint: {{constraint}}`),
		ai.WithOutputType([]string{}))

	type menuPlannerInput struct {
		NbMeals    int    `jsonschema_description:"The number of meals to create"`
		Constraint string `jsonschema_description:"The constraint to respect"`
	}

	menuPlannerFlow := genkit.DefineFlow(g, "menuPlannerFlow", func(ctx context.Context, in menuPlannerInput) (string, error) {
		menuRes, err := genkit.LookupPrompt(g, "menuPrompt").Execute(ctx,
			ai.WithInput(map[string]interface{}{
				"nb":         in.NbMeals,
				"constraint": in.Constraint,
			}))
		if err != nil {
			return "", err
		}
		var menus []string
		if err := menuRes.Output(&menus); err != nil {
			return "", err
		}

		fmt.Println("Menus to create:")
		for _, m := range menus {
			fmt.Println(m)
		}

		for _, menu := range menus {
			fmt.Println("Creating recipe for menu:", menu)
			recipe, err := createRecipe.Run(ctx, menu)
			if err != nil {
				return "", err
			}
			fmt.Println(recipe.String())
			fmt.Println(groceryList.String())
			res, err := groceryListManagerFlow.Run(ctx, recipe.Ingredients)
			if err != nil {
				return "", err
			}
			fmt.Println(res)
		}

		return "Done", nil
	})

	res, err := menuPlannerFlow.Run(ctx,
		menuPlannerInput{
			NbMeals:    2,
			Constraint: "something healthy for the winter",
		})

	if err != nil {
		panic(err)
	}
	fmt.Printf("%s\n", res)

	fmt.Println("# Current grocery list:")
	fmt.Println(groceryList.String())
}
