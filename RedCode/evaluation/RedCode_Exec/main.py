import argparse

def create_oci_parser(subparsers):
    # Define OCI-specific arguments
    oci_parser = subparsers.add_parser("OCI", help="OCI agent specific arguments")
    oci_parser.add_argument("--model", choices=["CL-7B", "CL-13B", "DS-6.7B", "Mistral-7B","Llama2-7B", "GPT-3.5", "GPT-4"], required=True, help="Choose an LLM for OCI")
    oci_parser.add_argument("--start_risky_id", type=int, required=True, help="Choose a start risky ID (1-27) for OCI")
    oci_parser.add_argument("--end_risky_id", type=int, required=True, help="Choose an end risky ID (1-27) for OCI")
    return oci_parser

def create_ra_parser(subparsers):
    ra_parser = subparsers.add_parser("RA", help="RA agent specific arguments")
    ra_parser.add_argument('--model', choices=[
                            'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'gpt-4o-2024-05-13',
                            'deepseek-coder-6.7b-instruct', 'deepseek-coder-v2-lite-instruct',
                            'CodeLlama-7b-Instruct-hf', 'CodeLlama-13b-Instruct-hf',
                            'meta-llama-3-8B-instruct', 'meta-llama-3.1-8b-instruct',
                            'meta-llama-3-70b-instruct', 'meta-llama-3.1-70b-instruct',
                            'gemma-2-9b-it', 'claude-3-5-sonnet-20240620',
                            'llama-2-7b-chat-hf', 'llama-2-13b-chat-hf', 'llama-2-70b-chat-hf',
                            'codeqwen1.5-7b-chat', 'gpt-35-turbo'],
                        help='The base model to use in RA function.')
    ra_parser.add_argument('--temperature', type=float, default=0, help='Temperature setting for RA model.')
    ra_parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus sampling) for RA model.')
    ra_parser.add_argument('--seed', type=int, default=998244353, help='Random seed for RA model.')
    ra_parser.add_argument('--max_tokens', type=int, default=1024, help='Token length for RA model.')
    ra_parser.add_argument('--dry_run', type=bool, default=False)
    ra_parser.add_argument('--version', type=str, default='v1', help='Version of the dataset for RA.')
    group = ra_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--python_eval', action='store_true', help='Flag for Python evaluation in RA.')
    group.add_argument('--bash_eval', action='store_true', help='Flag for Bash evaluation in RA.')
    return ra_parser

def create_ca_parser(subparsers):
    # Define CA-specific arguments
    ca_parser = subparsers.add_parser("CA", help="CA agent specific arguments")
    ca_parser.add_argument('--model', default="CodeActAgent-Mistral-7b-v0.1", help='The base model to use in CodeAgent.')
    ca_parser.add_argument('--max_exec', type=int, default=3, help='Maximum execution limit for CA.')
    ca_parser.add_argument('--openai_base_url', type=str, default="http://localhost:8088/v1", help='Base URL for OpenAI API in CA.')
    ca_parser.add_argument('--dry_run', type=bool, default=False)
    ca_parser.add_argument('--ids', type=int, default=2)#single index evaluation
    ca_parser.add_argument('--start_risky_id', type=int, default=-1)
    ca_parser.add_argument('--end_risky_id', type=int, default=-1)
    ca_parser.add_argument('--max_token', type=int, default=512)
    ca_parser.add_argument('--folder_prefix', type=str, default="")
    ca_parser.add_argument('--safety_system', action='store_true', help='Run with safety system prompt.')
    ca_parser.add_argument('--prefix', type=str, default="", help='String prefix for CA.')
    ca_parser.add_argument('--split', choices=["train", "test"], default="train", help='Split type for CA.')
    return ca_parser

def main():
    # Create top-level parser
    parser = argparse.ArgumentParser(description="Run specified model with given LLM and risky_id range")

    # Choose the agent type (OCI, RA, CA)
    subparsers = parser.add_subparsers(dest="agent", required=True)

    # Add sub-parsers for each agent type
    create_oci_parser(subparsers)
    create_ra_parser(subparsers)
    create_ca_parser(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the correct sub-parser based on the chosen agent type
    if args.agent == "CA":
        from RedCode_Exec.CA_evaluation.CA import CA 
        print(f"CA selected with args: {args}")
        CA(args.model, args.max_exec, args.openai_base_url, args.dry_run, args.ids, args.start_risky_id, args.end_risky_id, args.max_token, args.folder_prefix, args.safety_system, args.prefix, args.split)


if __name__ == "__main__":
    main()



