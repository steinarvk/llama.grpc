from proto import llama_pb2_grpc
from proto import llama_pb2

from absl import app
from absl import flags
from absl import logging

import grpc
import math
import sys
import random

FLAGS = flags.FLAGS
flags.DEFINE_string("server", "localhost:50051", "Address of server to connect to.")
flags.DEFINE_float("temperature", 1.0, "Temperature for token generation")
flags.DEFINE_string("model_name", "65B/ggml-model-q4_0", "Model to use")
flags.DEFINE_integer("max_tokens", 0, "Number of tokens to stop after")

def choose_softmax(logits, temperature=0.5):
    choices = [(math.exp(logit.logit / temperature), logit.token) for logit in logits]
    total = sum([choice[0] for choice in choices])
    x = random.random() * total
    for choice in choices:
        x -= choice[0]
        if x <= 0:
            break
    return choice[1]

def main(argv):
    del argv

    with grpc.insecure_channel(FLAGS.server) as channel:
        stub = llama_pb2_grpc.LlamaServiceStub(channel)

        prompt = sys.stdin.read()

        req = llama_pb2.DoPredictRequest()
        req.model_info.model_name = FLAGS.model_name
        req.full_context.str = prompt
        req.logit_processing.top_n = 40
        req.logit_processing.llama_repetition_penalty.intensity = 1.1

        max_tokens = FLAGS.max_tokens
        if max_tokens == 0:
            max_tokens = 100000

        def step():
            response = stub.DoPredict(req)
            req.session_hint.session_id = response.session_info.session_id
            req.full_context.token_ids.CopyFrom(response.full_input_context)

            chosen_token = choose_softmax(response.next_token_logit, temperature=FLAGS.temperature)
            if chosen_token.token_id <= 2:
                return False

            req.full_context.token_ids.token_id.append(chosen_token.token_id)

            sys.stdout.write(chosen_token.token_str.decode("utf-8"))
            sys.stdout.flush()

            return True

        for _ in range(max_tokens):
            if not step():
                break

if __name__ == '__main__':
  app.run(main)