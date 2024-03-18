import { ChatOpenAI } from "@langchain/openai";
import {ChatPromptTemplate} from "@langchain/core/prompts"

import * as dotenv from "dotenv";
dotenv.config();

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7 //1 - fully creative, 0 - strict and factual
});

//const prompt = ChatPromptTemplate.fromTemplate('You are a comedian. Tell a joke based on the following word {input}')
const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Generate a joke based on a word provided by the user."],
    ["human", "{input}"]
])
//console.log(await prompt.format({input: 'chicken' }))

// create chain
const chain = prompt.pipe(model);

// call chain
const resp = await chain.invoke({
    input: 'dog'
})

console.log(resp)