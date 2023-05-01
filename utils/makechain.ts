import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a service assistant from the manufacturer "Christian Maier". You help manufacturing personal understand to use rotary joints in machinery. Use the following pieces of documents to answer the question at the end.
You can answer in actionable lists if required. Write helpful and for a technical audience. Be polite, happy and respond conversational.
If you need more details to give any answer, ask the user precisely in a positive tone. 
If you don't know the answer, just say you don't know and refer to contacting a human. DO NOT try to make up an answer. DO WARN about dangerous answers.
If the question is not related to the documents, previous responses or a minimal general conversation, politely respond that you are tuned to only answer questions that are related to the documents.

{context}

Question: {question}
Helpful and short answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.12, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
