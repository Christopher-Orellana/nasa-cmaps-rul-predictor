from src.inference.predict import predict_rul

def main():
    # Example single-engine snapshot
    example_input = {
        "sensor_4": 1410.5,
        "sensor_11": 47.6,
        "sensor_15": 8.45
    }

    output = predict_rul(example_input)
    print("Inference output")
    print(output)

if __name__ == "__main__":
    main()