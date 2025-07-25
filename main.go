package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores/weaviate"
)

const (
	modelEmbedding = "nomic-embed-text"
	modelLLM       = "mistral"
	promptTemplate = `Answer the question based only on the following context:

%v

---

Answer the question based on the above context: %v`
)

func main() {
	ctx := context.Background()

	// 1. Read the markdown file
	data, err := os.ReadFile("data.md")
	if err != nil {
		panic(err)
	}
	text := string(data)

	// 2. Split the document into chunks
	splitter := textsplitter.NewRecursiveCharacter()
	chunks, err := splitter.SplitText(text)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Received %d chunks\n", len(chunks))

	// 3. Initialize the embedder and Weaviate vector store
	embedder := NewOllamaEmbedder(modelEmbedding, "http://localhost:11434")

	store, err := weaviate.New(
		weaviate.WithIndexName("Docs"),
		weaviate.WithScheme("http"),
		weaviate.WithHost("localhost:8080"),
		weaviate.WithEmbedder(embedder))
	if err != nil {
		panic(err)
	}

	// 4. Convert chunks into schema.Documents and add to Weaviate
	docs := make([]schema.Document, 0, len(chunks))
	for _, chunk := range chunks {
		docs = append(docs, schema.Document{
			PageContent: chunk,
			Metadata:    map[string]any{}, // optional metadata
		})
	}

	if _, err = store.AddDocuments(ctx, docs); err != nil {
		panic(err)
	}

	// 5. Define the user question
	userQuestion := "How many chapters in the Alice’s Adventures in Wonderland?"

	// 6. Perform similarity search for top-3 relevant documents
	relevant, err := store.SimilaritySearch(ctx, userQuestion, 3)
	if err != nil {
		panic(err)
	}
	var contextText []string
	for _, doc := range relevant {
		contextText = append(contextText, doc.PageContent)
	}

	// 7. Construct the final prompt
	finalPrompt := fmt.Sprintf(promptTemplate, strings.Join(contextText, "\n---\n"), userQuestion)
	log.Println(finalPrompt)

	// 8. Query the LLM with the prompt
	llm, err := ollama.New(ollama.WithModel(modelLLM))
	if err != nil {
		panic(err)
	}
	completion, err := llm.Call(ctx, finalPrompt)
	if err != nil {
		panic(err)
	}

	fmt.Println("\nAnswer from the model:")
	fmt.Println(completion)
}
