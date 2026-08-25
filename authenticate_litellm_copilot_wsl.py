from pathlib import Path

from litellm.llms.github_copilot.authenticator import Authenticator


def main() -> None:
    authenticator = Authenticator()
    info = authenticator._get_device_code()
    print(f"Open: {info['verification_uri']}", flush=True)
    print(f"Code: {info['user_code']}", flush=True)
    access_token = authenticator._poll_for_access_token(info["device_code"])

    access_path = Path(authenticator.access_token_file)
    access_path.write_text(access_token, encoding="utf-8")
    access_path.chmod(0o600)
    authenticator.get_api_key()
    Path(authenticator.api_key_file).chmod(0o600)
    print("GitHub Copilot OAuth initialized for LiteLLM.")


if __name__ == "__main__":
    main()