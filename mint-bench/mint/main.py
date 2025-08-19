from mint.envs import GeneralEnv, AlfworldEnv
from mint.datatypes import Action, State
from mint.tasks import AlfWorldTask
from mint.tools import Tool
import mint.tasks as tasks
import mint.agents as agents
import logging
import os
import json
import pathlib
import importlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from mint.agents.vllm_agent import UnsafePromptError

test_indexes = []

# Configure logging settings
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("MINT")


def interactive_loop(
    task: tasks.Task,
    agent: agents.LMAgent,
    tools: List[Tool],
    feedback_config: Dict[str, Any],
    env_config: Dict[str, Any],
    interactive_mode: bool = False,
    safety_prompt: bool = False,
):
    if isinstance(task, AlfWorldTask):
        LOGGER.info("loading Alfworld Env")
        env = AlfworldEnv(task, tools, feedback_config, env_config, safety_prompt=safety_prompt)
    else:
        env = GeneralEnv(task, tools, feedback_config, env_config, safety_prompt=safety_prompt)
    state: State = env.reset()

    init_msg = state.latest_output['content']
    if interactive_mode:
        # omit in-context example
        splited_msg = init_msg.split("---")
        init_msg = splited_msg[0] + "== In-context Example Omitted ==" + splited_msg[2]

    LOGGER.info(f"\nUser: \n\033[94m{state.latest_output['content']}\033[0m")

    num_steps = 0

    if task.loaded_history is not None:
        for turn in task.loaded_history:
            action = agent.lm_output_to_action(turn["lm_output"])
            LOGGER.info(
                f"\nLoaded LM Agent Action:\n\033[92m{action.value}\033[0m")
            state = env.step(action, loaded=turn)
            LOGGER.info(
                "\033[1m" + "User:\n" + "\033[0m" +
                f"\033[94m{state.latest_output['content']}\033[0m"
            )
            num_steps += 1

    while not state.finished:
        # agent act
        if interactive_mode:
            to_continue = "n"
            while to_continue not in ["y", "Y"]:
                to_continue = input("\n> Continue? (y/n) ")

        try:
            action: Action = agent.act(state)
        except UnsafePromptError:
            LOGGER.warning("Prompt unsafe. Marking task as finished and failed.")
            state.success = False
            state.finished = True
            break
        # color the action in green
        # LOGGER.info(f"\nLM Agent Action:\n\033[92m{action.value}\033[0m")
        LOGGER.info(
            f"\n\033[1m" + "LM Agent Action:\n" + "\033[0m" +
            f"\n\033[92m{action.value}\033[0m"
        )
        # environment step
        state: State = env.step(action)
        # color the state in blue
        if not state.finished:
            user_msg = state.latest_output['content']
            if "Expert feedback:" in user_msg:
                try:
                    splited = user_msg.split("Expert feedback:")
                    obs, feedback = splited[0], "".join(splited[1:])
                except Exception as e:
                    LOGGER.info(f"Error: {e}")
                    LOGGER.info(f"User message: {user_msg}")
                    LOGGER.exception(e)
                    raise e

                feedback = "Expert feedback:" + feedback
                # color the observation in blue & feedback in red
                LOGGER.info(
                    "\n" +
                    "\033[1m" + "User:\n" + "\033[0m" +
                    f"\033[94m{obs}\033[0m" + "\n" 
                    + f"\033[93m{feedback}\033[0m" + "\n"
                )
            else:
                # color the observation in blue
                LOGGER.info(
                    "\n" +
                    "\033[1m" + "User:\n" + "\033[0m" +
                    f"\033[94m{user_msg}\033[0m" + "\n"
                )
        num_steps += 1

    LOGGER.info(
        f"Task finished in {num_steps} steps. Success: {state.success}"
    )

    return state


def main(args: argparse.Namespace):
    with open(args.exp_config) as f:
        exp_config: Dict[str, Any] = json.load(f)

    DEFAULT_FEEDBACK_CONFIG = exp_config["feedback_config"]
    DEFAULT_ENV_CONFIG = exp_config["env_config"]

    LOGGER.info(f"Experiment config: {exp_config}")

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    todo_tasks, n_tasks = task_class.load_tasks(task_config["filepath"])

    # initialize the agent
    agent_config: Dict[str, Any] = exp_config["agent"]
    agent_config_dict = agent_config["config"]
    if args.prefix:
        agent_config_dict["prefix"] = args.prefix
    agent_config_dict["use_guard"] = args.guard
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        #agent_config["config"]
        agent_config_dict
    )
    agent.guard_checked = False

    # initialize the feedback agent (if exist)
    feedback_config: Dict[str, Any] = exp_config.get(
        "feedback", DEFAULT_FEEDBACK_CONFIG
    )

    # initialize all the tools
    tools: List[Tool] = [
        getattr(importlib.import_module(module), class_name)()
        for module, class_name in task_config["tool_imports"]
    ]

    env_config: Dict[str, Any] = exp_config.get(
        "environment", DEFAULT_ENV_CONFIG)

    pathlib.Path(exp_config["output_dir"]).mkdir(parents=True, exist_ok=True)
    if args.interactive:
        output_path = os.path.join(
            exp_config["output_dir"], "results.interactive.jsonl")
    else:
        output_path = os.path.join(exp_config["output_dir"], "results.jsonl")

    done_task_id = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                task_id = json.loads(line)["task"].get("task_id", "")
                if task_id == "":
                    task_id = json.loads(line)["task"].get("id", "")
                done_task_id.add(task_id)
        LOGGER.info(
            f"Existing output file found. {len(done_task_id)} tasks done.")

    if len(done_task_id) == n_tasks:
        LOGGER.info("All tasks done. Exiting.")
        return

    # run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_tasks = n_tasks - len(done_task_id)  # only run the remaining tasks
    with open(output_path, "a") as f, logging_redirect_tqdm():
        if args.split == "test":
            pbar = tqdm(total=10)
        else:
            pbar = tqdm(total=n_tasks-10)
        for i, task in enumerate(todo_tasks):
            if args.split == "train" and i in test_indexes:
                continue
            if args.split == "test" and i not in test_indexes:
                continue
            # Only test 10 tasks in debug mode
            if args.debug and i == 3:
                break

            # skip done tasks
            if task.task_id in done_task_id:
                continue

            state = interactive_loop(
                task, agent, tools, feedback_config, env_config, args.interactive, args.safety_prompt
            )
            agent.guard_checked = False 
            if not os.path.exists(exp_config["output_dir"]):
                os.makedirs(exp_config["output_dir"])
            f.write(
                json.dumps({"state": state.to_dict(),
                           "task": task.to_dict()}) + "\n"
            )
            f.flush()  # make sure the output is written to file
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_config",
        type=str,
        default="./configs/gpt-3.5-turbo-0613/F=gpt-3.5-turbo-16k-0613/PHF=GT-textual/max5_p2+tool+cd/reasoning/scienceqa.json",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="prefix",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="split",
    )
    parser.add_argument(
        "--guard",
        action="store_true",
        help="Run Llama-Guard safety checker at first step.",
    )
    parser.add_argument(
        "--safety_prompt",
        action="store_true",
        help="Whether to use the safe prompt template.",
    )

    args = parser.parse_args()
    LOGGER.setLevel(logging.DEBUG if args.debug else logging.INFO)
    main(args)
