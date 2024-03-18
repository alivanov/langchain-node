import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
dotenv.config();

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7, //1 - fully creative, 0 - strict and factual
    maxTokens: 1000,
    verbose: true
});

const resp = await model.invoke('Hello');
//const resp = await model.batch(['Hello', "How are you?"]);
//const resp = await model.stream('write a poem about AI');
//const resp = await model.streamLog('write a poem about AI');

console.log(resp)

/* for await (const chunk of resp) {
    //console.log(chunk?.content)
    console.log(chunk)
} */
