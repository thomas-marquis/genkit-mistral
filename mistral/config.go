package mistral

import (
	"github.com/thomas-marquis/mistral-client/mistral"
)

type Config struct {
	Client mistral.Client
}

type Option func(*Config)

func WithClient(client mistral.Client) Option {
	return func(c *Config) {
		c.Client = client
	}
}

func NewConfig(opts ...Option) *Config {
	c := &Config{}
	for _, opt := range opts {
		opt(c)
	}
	return c
}
