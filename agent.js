import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts"
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents"

import { TavilySearchResults } from "@langchain/community/tools/tavily_search"

import readline from "readline";

import { AIMessage, HumanMessage } from "@langchain/core/messages"

import * as dotenv from "dotenv";
dotenv.config();

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
const searchTool = new TavilySearchResults()
const tools = [searchTool];

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