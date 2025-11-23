package mistralclient

type Role string

func (r Role) String() string {
	return string(r)
}

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleTool      Role = "tool"
)
