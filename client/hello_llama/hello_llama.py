from proto import llama_pb2_grpc
from proto import llama_pb2
import grpc

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = llama_pb2_grpc.LlamaServiceStub(channel)

        stub.DoLoadModel(llama_pb2.DoLoadModelRequest(
            model_name="13B",
        ))

        response = stub.Tokenize(llama_pb2.TokenizeRequest(
            text = "Hello world, this is just a test!",
        ))
        for token in response.token:
            print(token.token_id, token.token_str)


if __name__ == '__main__':
    run()
