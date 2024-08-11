from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


res = client.embeddings.create(
 input="The Godfather",
 model="text-embedding-ada-002"
  
)
print(res.data[0].embedding)
