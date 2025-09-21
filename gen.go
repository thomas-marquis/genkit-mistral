package main

import _ "go.uber.org/mock/gomock"

//go:generate mockgen -package mocks -destination mocks/client.go github.com/thomas-marquis/genkit-mistral/mistralclient Client
