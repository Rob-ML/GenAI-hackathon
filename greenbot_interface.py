# Databricks notebook source
# MAGIC %pip install imageio diffusers opencv-python

# COMMAND ----------

# MAGIC %pip install databricks-genai-inference databricks-vectorsearch 
# MAGIC dbutils.library.restartPython() 
# MAGIC

# COMMAND ----------

query = dbutils.widgets.get("Prompt")

# COMMAND ----------

CATALOG = "workspace"
DB='default'

# Create the Vector Index
VS_INDEX_NAME = 'dais_hackathon'
VS_INDEX_FULLNAME = f"{CATALOG}.{DB}.{VS_INDEX_NAME}"

VS_ENDPOINT_NAME = 'genai_endpoint'


# COMMAND ----------

# MAGIC %md
# MAGIC #### Get index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME,
                      index_name=VS_INDEX_FULLNAME)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run inference

# COMMAND ----------


from databricks_genai_inference import ChatSession

chat = ChatSession(model="databricks-meta-llama-3-70b-instruct",
                   system_message="You are a helpful assistant.",
                   max_tokens=128)

# COMMAND ----------

# reset history
chat = ChatSession(model="databricks-meta-llama-3-70b-instruct",
                   system_message="You are a helpful assistant. Answer the user's question based on the provided context. Explain it in a very simple terms, use analogies if relevant. The goal of the output is to provide a script for a short video. Your answer needs to be very short when it is complete, it needs to be less that 80 words",
                   max_tokens=80)

# get context from vector search
raw_context = index.similarity_search(columns=["text"],
                        query_text=query,
                        num_results = 3)

context_string = "Context:\n\n"

for (i,doc) in enumerate(raw_context.get('result').get('data_array')):
    context_string += f"Retrieved context {i+1}:\n"
    context_string += doc[0]
    context_string += "\n\n"

chat.reply(f"User question: {query}\n\nContext: {context_string}")
first_step_output = chat.last

# COMMAND ----------

# reset history
chat = ChatSession(model="databricks-meta-llama-3-70b-instruct",
                   system_message="You are an expert visualiser. The goal of the output is to provide a prompt for a video generation engine. The prompt should be extremely simple, understandable even by children, no matter how complex the topic is short - less than 80 tokens. Skip any intro like Here is a simplified prompt for a video generation engine or similar, jump straight to the description.",
                   max_tokens=80)

chat.reply(f"Please simplify this text to create a video prompt: {first_step_output}. Imagine you were prompting to draw a nice picture representing the text that we want to simplify.")
final_prompt = chat.last


# COMMAND ----------

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.enable_model_cpu_offload()

# memory optimization
pipe.enable_vae_slicing()

prompt = chat.last
video_frames = pipe(final_prompt, num_frames=64).frames[0]
video_path = export_to_video(video_frames)
video_path

# COMMAND ----------

# Move the file to dbfs for local download

original = f'file:{video_path}'
new = f"dbfs:/tmp/videos_demo/video.mp4"

dbutils.fs.cp(original, new)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## HOW TO download to local environment:
# MAGIC
# MAGIC - Set up databricks cli
# MAGIC - run comment: `databricks fs cp dbfs:/tmp/videos_demo/video.mp4 [desired local path] `

# COMMAND ----------


