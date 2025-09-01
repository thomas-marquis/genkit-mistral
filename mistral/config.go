package mistral

import (
	"github.com/thomas-marquis/genkit-mistral/mistralclient"
)

type Config struct {
	Client mistralclient.Config
}

type Option func(*Config)

func WithClientConfig(cfg mistralclient.Config) Option {
	return func(c *Config) {
		c.Client = cfg
	}
}

func NewConfig(opts ...Option) *Config {
	c := &Config{}
	for _, opt := range opts {
		opt(c)
	}
	return c
}
