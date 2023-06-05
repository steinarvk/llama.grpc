from proto import llama_pb2_grpc
from proto import llama_pb2
import grpc

SAMPLE = """
Hello world, this is just a test:

Foo: bar
Baz: quux
"""

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = llama_pb2_grpc.LlamaServiceStub(channel)

        stub.DoLoadModel(llama_pb2.DoLoadModelRequest(
            model_name="13B",
        ))

        response = stub.GetVocabulary(llama_pb2.GetVocabularyRequest())
        for token in response.token:
            print(token.token_id, repr(token.token_str))

        response = stub.Tokenize(llama_pb2.TokenizeRequest(
            text = SAMPLE
        ))
        for token in response.token:
            print(token.token_id, repr(token.token_str))


if __name__ == '__main__':
    run()
