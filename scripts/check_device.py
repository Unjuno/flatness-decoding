from flatness_decoding.devices import describe_torch_devices, resolve_device


def main() -> None:
    info = describe_torch_devices()
    for key, value in info.items():
        print(f"{key}: {value}")
    print(f"selected_device_auto: {resolve_device('auto')}")


if __name__ == "__main__":
    main()
