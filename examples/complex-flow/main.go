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

	groceryList := struct {
		sync.Mutex
		indexedList map[int]string
		lastIndex   int
	}{
		indexedList: make(map[int]string),
	}

	genkit.DefineTool(g, "groceryListGet", "get the current content of the list", func(ctx *ai.ToolContext, input any) (string, error) {
		if len(groceryList.indexedList) == 0 {
			return "The list is empty", nil
		}
		rendered := strings.Builder{}
		for i, item := range groceryList.indexedList {
			rendered.WriteString("- ")
			rendered.WriteString(fmt.Sprint(i + 1))
			rendered.WriteString(item)
			rendered.WriteString("\n")
		}
		return rendered.String(), nil
	})

	type groceryListAddInput struct {
		Item string `jsonschema_description:"The item to add to the list and the quantity indication"`
	}

	genkit.DefineTool(g, "groceryListAdd", "add an item to the list", func(ctx *ai.ToolContext, input groceryListAddInput) (string, error) {
		fmt.Printf("groceryListAdd: %s\n", input.Item)
		groceryList.Lock()
		defer groceryList.Unlock()
		groceryList.indexedList[groceryList.lastIndex+1] = input.Item
		groceryList.lastIndex++
		return fmt.Sprintf("Item %d added to the list", groceryList.lastIndex), nil
	})

	type groceryListDeleteInput struct {
		Index int `jsonschema_description:"The index of the item to remove from the list"`
	}

	genkit.DefineTool(g, "groceryListDelete", "remove an item from the grocery list", func(ctx *ai.ToolContext, input groceryListDeleteInput) (string, error) {
		fmt.Printf("groceryListDelete: %d\n", input.Index)
		groceryList.Lock()
		defer groceryList.Unlock()
		delete(groceryList.indexedList, input.Index)
		return fmt.Sprintf("Item %d removed from the list", input.Index), nil
	})

	type groceryListUpdateInput struct {
		Index int    `jsonschema_description:"The index of the item to update in the list"`
		Item  string `jsonschema_description:"The new item to replace the old one with the updated label and/or quantity indication"`
	}

	genkit.DefineTool(g, "groceryListUpdate", "update an item in the grocery list", func(ctx *ai.ToolContext, input groceryListUpdateInput) (string, error) {
		fmt.Printf("groceryListUpdate: %d %s\n", input.Index, input.Item)
		groceryList.Lock()
		defer groceryList.Unlock()
		groceryList.indexedList[input.Index] = input.Item
		return fmt.Sprintf("Item %d updated in the list", input.Index), nil
	})

	genkit.DefinePrompt(g, "groceryListPrompt",
		ai.WithSystem(`You are a grocery list manager. Your role is to keep it as clean and coherent as possible. 
		Avoid duplication and ensure you don't forget any item on the list.`),
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

	menuPlannerFlow := genkit.DefineFlow(g, "menuPlannerFlow", func(ctx context.Context, in []string) (string, error) {
		for _, menu := range in {
			recipe, err := createRecipe.Run(ctx, menu)
			if err != nil {
				return "", err
			}
			fmt.Println(recipe.String())
			res, err := groceryListManagerFlow.Run(ctx, recipe.Ingredients)
			if err != nil {
				return "", err
			}
			fmt.Println(res)
		}

		return "Done", nil
	})

	res, err := menuPlannerFlow.Run(ctx,
		[]string{
			"Make a pizza with cheese and tomatoes",
			"Make a salad with lettuce and tomatoes",
		})

	if err != nil {
		panic(err)
	}
	fmt.Printf("%s\n", res)

	fmt.Println("# Current grocery list:")
	for _, item := range groceryList.indexedList {
		fmt.Println(item)
	}
}
