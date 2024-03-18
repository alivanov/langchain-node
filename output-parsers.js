import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { StringOutputParser, CommaSeparatedListOutputParser } from "@langchain/core/output_parsers"
import { StructuredOutputParser } from "langchain/output_parsers"
import { z } from "zod"

import * as dotenv from "dotenv";
dotenv.config();

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7 //1 - fully creative, 0 - strict and factual
});

async function callStringParser() {
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Generate a joke based on a word provided by the user."],
        ["human", "{input}"]
    ])
    
    const parser = new StringOutputParser()
    
    // create chain
    const chain = prompt.pipe(model).pipe(parser);
    
    // call chain
    return await chain.invoke({
        input: 'dog'
    })
}

async function callListOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate("Provide 5 synonims, separated by commas, for the following word {input}")
    const parser = new CommaSeparatedListOutputParser()
    const chain = prompt.pipe(model).pipe(parser);
    
    return await chain.invoke({
        input: 'happy'
    })
}

async function callListStructuredParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
        Extract information from the following phrase.
        Formatting Instructions: {format_instructions}
        Phrase: {input}
    `)
    const parser = StructuredOutputParser.fromNamesAndDescriptions({
        name: "the name of the person",
        age: "the age of the person"
    })
    const chain = prompt.pipe(model).pipe(parser);
    
    return await chain.invoke({
        input: 'Max is 30 years old',
        format_instructions: parser.getFormatInstructions()
    })
}

async function callZodOutputParser() {
    const prompt = ChatPromptTemplate.fromTemplate(`
        Extract information from the following phrase.
        Formatting Instructions: {format_instructions}
        Phrase: {input}
    `)
    const parser = StructuredOutputParser.fromZodSchema(
        z.object({
            recipe: z.string().describe("name of recipe"),
            ingredients: z.array(z.string()).describe("ingredients")
        }))
    const chain = prompt.pipe(model).pipe(parser);
    
    return await chain.invoke({
        input: 'The ingredience for Spaghetti Bolognese recipe are tomatoes, minced beef, garlic, wine and herbs.',
        format_instructions: parser.getFormatInstructions()
    })
}

//const resp = await callStringParser()
//const resp = await callListOutputParser()
//const resp = await callListStructuredParser()
const resp = await callZodOutputParser()
console.log(resp)