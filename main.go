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

	// 1. Читаем markdown-файл
	data, err := os.ReadFile("data.md")
	if err != nil {
		panic(err)
	}
	text := string(data)

	// 2. Разбиваем на чанки
	splitter := textsplitter.NewRecursiveCharacter()
	chunks, err := splitter.SplitText(text)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Recieved %d chunks\n", len(chunks))

	// 3. Создаём embedding-модель и Weaviate store
	embedder := NewOllamaEmbedder(modelEmbedding, "http://localhost:11434")

	store, err := weaviate.New(
		weaviate.WithIndexName("Docs"),
		weaviate.WithScheme("http"),
		weaviate.WithHost("localhost:8080"),
		weaviate.WithEmbedder(embedder))
	if err != nil {
		panic(err)
	}

	docs := make([]schema.Document, 0, len(chunks))
	for _, chunk := range chunks {
		docs = append(docs, schema.Document{
			PageContent: chunk,
			Metadata:    map[string]any{}, // можно добавить информацию, например {"source": "data.md"}
		})
	}

	// 4. Сохраняем чанки в Weaviate
	if _, err = store.AddDocuments(ctx, docs); err != nil {
		panic(err)
	}

	// 5. Получаем вопрос пользователя
	userQuestion := "How many chapters in the Alice’s Adventures in Wonderland?"

	// 6. Ищем релевантные чанки (топ-3)
	relevant, err := store.SimilaritySearch(ctx, userQuestion, 3)
	if err != nil {
		panic(err)
	}
	var contextText []string
	for _, doc := range relevant {
		contextText = append(contextText, doc.PageContent)
	}

	// 7. Формируем prompt
	finalPrompt := fmt.Sprintf(promptTemplate, strings.Join(contextText, "\n---\n"), userQuestion)

	log.Println(finalPrompt)

	// 8. Отправляем в LLM
	llm, err := ollama.New(ollama.WithModel(modelLLM))
	if err != nil {
		panic(err)
	}
	completion, err := llm.Call(ctx, finalPrompt)
	if err != nil {
		panic(err)
	}

	fmt.Println("\n✅ Ответ модели:")
	fmt.Println(completion)
}
