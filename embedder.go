package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/tmc/langchaingo/embeddings"
)

type OllamaEmbedder struct {
	Model string
	Host  string // e.g., http://localhost:11434
}

func NewOllamaEmbedder(model string, host string) *OllamaEmbedder {
	return &OllamaEmbedder{
		Model: model,
		Host:  host,
	}
}

func (e *OllamaEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, 0, len(texts))
	for _, text := range texts {
		vec, err := e.embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("EmbedDocuments error: %w", err)
		}
		embeddings = append(embeddings, vec)
	}
	return embeddings, nil
}

func (e *OllamaEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return e.embed(ctx, text)
}

func (e *OllamaEmbedder) embed(ctx context.Context, prompt string) ([]float32, error) {
	body := map[string]string{
		"model":  e.Model,
		"prompt": prompt,
	}
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.Host+"/api/embeddings", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding request failed: %s", b)
	}

	var response struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, errors.New("failed to decode embedding response")
	}
	return response.Embedding, nil
}

var _ embeddings.Embedder = (*OllamaEmbedder)(nil) // compile-time check
