import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts"
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents"
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { TavilySearchResults } from "@langchain/community/tools/tavily_search"
import { createRetrieverTool } from "langchain/tools/retriever"

import readline from "readline";

import { AIMessage, HumanMessage } from "@langchain/core/messages"

import * as dotenv from "dotenv";
dotenv.config();



const loader = new CheerioWebBaseLoader("https://js.langchain.com/docs/expression_language");
const docs= await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20
})

const splitDocs = await splitter.splitDocuments(docs)

const embeddings = new OpenAIEmbeddings()
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings)

//Retrieve data
const retriever = vectorStore.asRetriever({
    k: 2
});


const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106",
    temperature: 0.7 //1 - fully creative, 0 - strict and factual
});

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant called Max"],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad")
]);

// Create & Assign Tools
const searchTool = new TavilySearchResults();
const retrieverTool = createRetrieverTool(retriever, {
    name: "lcel_search",
    description: "Use this tool when searching for information about Lanchain Expression Language (LCEL)"
});
const tools = [searchTool, retrieverTool];

// Create Agent
const agent = await createOpenAIFunctionsAgent({
    llm: model,
    prompt,
    tools
});

// Create Agent Executor
const agentExecutor = new AgentExecutor({
    agent,
    tools
});

/*
// Call Agent
const resp = await agentExecutor.invoke({
    input: "What is the current weather in Cape Town, South Africa?"
});

console.log(resp) */

// Get user input
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const chatHistory = [];


const askQuestion = () => {
    rl.question("User: ", async (input) => {

        if (input.toLowerCase() === 'exit' || input.toLowerCase() === 'quit') {
            rl.close();
            return;
        }

        // Call Agent
        const resp = await agentExecutor.invoke({
            input,
            chat_history: chatHistory
        });
    
        console.log("Agent: ", resp.output);

        chatHistory.push(new HumanMessage(input));
        chatHistory.push(new AIMessage(resp.output));

        askQuestion();
    })
}

askQuestion();