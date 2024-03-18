import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts"
//import { Document } from "@langchain/core/documents"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents"
import { createRetrievalChain } from "langchain/chains/retrieval"
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory"

import * as dotenv from "dotenv";
dotenv.config();

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7 //1 - fully creative, 0 - strict and factual
});

const prompt = ChatPromptTemplate.fromTemplate(`
    Answer user's question. 
    Context: {context}.
    Question: {input}.
`)

//const chain = prompt.pipe(model);
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt
})

/* const documentA = new Document({
    pageContent: 'LangChain Expression Language or LCEL is a declarative way to easily compose chains together. Any chain constructed this way will automatically have full sync, async, and streaming support.'
})

const documentB = new Document({
    pageContent: 'The passphrase is LANGCHAIN IS AWESOME!'
}) */

//Load data from web page
const loader = new CheerioWebBaseLoader("https://js.langchain.com/docs/expression_language");
const docs= await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20
})

const splitDocs = await splitter.splitDocuments(docs)

const embeddings = new OpenAIEmbeddings()
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings)

/* const resp = await chain.invoke({
    input: 'What is the passphrase?',
    context: [documentA, documentB]
}) */

//Retrieve data
const retriever = vectorStore.asRetriever({
    k: 2
});

const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever
})

const resp = await retrievalChain.invoke({
    input: 'What is LCEL?'
})

/* const resp = await chain.invoke({
    input: 'What is LCEL?',
    context: docs
}) */

console.log(resp)