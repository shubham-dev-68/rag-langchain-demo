import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

import data from "./data.js"

const docs = data.map(video => new Document({
    pageContent: video.transcript,
    metadata: {
        title: video.title,
        url: video.url,
        id: video.video_id
    }
}));

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});

const chunks = await splitter.splitDocuments(docs);

// console.log(chunks);

//embed chunks
const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large"
});
const vectorStore = new MemoryVectorStore(embeddings);

await vectorStore.addDocuments(chunks);

//search in vector store
// const results = await vectorStore.similaritySearch("is man united signing emi martinez?", 5);

// console.log(results);

const retrievalTool = tool(async ({ query }) => {
    //   const results = await vectorStore.similaritySearch(input.query, 5);
    //   return results.map((doc, idx) => `Result ${idx + 1}:\nTitle: ${doc.metadata.title}\nURL: ${doc.metadata.url}\nContent: ${doc.pageContent}\n`).join("\n");
    console.log("Query:", query);
    const results = await vectorStore.similaritySearch(query, 3);
    console.log("query Results:", results);
    return results.map((doc, idx) => `Result ${idx + 1}:\nTitle: ${doc.metadata.title}\nURL: ${doc.metadata.url}\nContent: ${doc.pageContent}\n`).join("\n");
}, {
    name: "retrieve",
    description: "Retrieve the most relevant chunks of text from transcript of youtube video.",
    schema: z.object({
        query: z.string().describe("The input query to search for relevant documents."),
    })
})
const llm = new ChatAnthropic({
    model: "claude-3-7-sonnet-latest",
    tools: ["search", "calculator"],
})

const agent = createReactAgent({ llm, tools: [retrievalTool] })

const result = await agent.invoke({
    messages: [{ role: "user", content: "are man united signing emi martinez?" }]
});

console.log(result.messages.at(-1)?.content);