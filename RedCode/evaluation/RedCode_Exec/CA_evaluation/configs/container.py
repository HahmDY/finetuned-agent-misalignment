import docker
import json
import signal
import logging
import time
import os

class Container:
    def __init__(self, image, name=None):
        self.image = image
        curr_time = time.time()
        self.name = image + str(curr_time)
        self.client = docker.from_env()
        
        # Check if the image exists, and build it if not
        if not self.image_exists(self.image):
            logging.info(f"Image {self.image} not found. Building the image from Dockerfile.")
            self.build_image()
        
        # Create the container
        self.container = self.client.containers.run(
            self.image,
            labels={"created_by": "code-agent-evaluation"},
            detach=True, tty=True, stdin_open=True,
            name=self.name
        )
        print(f"Container {self.container} started successfully.")
        logging.info(f"Container {self.name} started successfully.")
        
        # Run initialization script
        output = self.container.exec_run("bash /app/init.sh")
        output_str = output.output.decode()
        logging.info(f"Container init output: {output_str}")

    def __del__(self):
        try:
            self.container.stop()
        except:
            pass
        finally:
            self.container.remove()

    def __enter__(self):
        
        try:
            logging.info(f"Starting container {self.name} in __enter__...")
            output = self.container.exec_run("bash /app/init.sh")
            output_str = output.output.decode()
            logging.info(f"Container init output: {output_str}")
        except Exception as e:
            logging.error(f"Failed to start container: {e}")
            self.container = None
        return self

    def image_exists(self, image_name):
        try:
            # Attempt to fetch the image
            self.client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            return False

    def build_image(self):
        try:
            dockerfile_path = os.path.join(os.path.dirname(__file__), "../../../../environment")
            logging.info(f"Building image {self.image} from Dockerfile in {dockerfile_path}.")
            self.client.images.build(path=dockerfile_path, tag=self.image)
            logging.info(f"Image {self.image} built successfully.")
        except Exception as e:
            logging.error(f"Failed to build image {self.image}: {e}")
            raise
    
    def execute_with_retries(self, cmd, retries=3, timeout=1*60):
        def handler(signum, frame):
            raise TimeoutError("Execution timed out")
        
        signal.signal(signal.SIGALRM, handler)
        
        for attempt in range(retries):

     
            try:
                logging.debug(f"Attempt {attempt+1}: Starting exec_run in {cmd}")
                signal.alarm(timeout)
                output = self.container.exec_run(cmd, stdin=False, stdout=True, stderr=True, demux=False)
                signal.alarm(0)
                output_str = output.output.decode()
                logging.debug("Finished exec_run")

                if "EOFError" in output_str:
                    logging.error(f"EOFError detected in command output: {cmd}\nOutput:\n{output_str}")
                    return "Execution blocked due to EOFError"
                
                return output_str
            except TimeoutError:
                logging.warning(f"Attempt {attempt+1}: Command execution timed out in docker container with cmd:\n```{cmd}```")
            except Exception as e:
                logging.error(f"Attempt {attempt+1}: An error occurred in docker container: {e}")
            
            logging.info(f"Retrying in 2 seconds...")
            time.sleep(2)
        
        return "Execution timed out"

    def execute_independent(self, command, *params):
        # print("=== EXECUTING INDEPENDENT ===\n", command)
        language, command = command
        # if params:
        #     print("== Parameters ==\n", params)
        if language == "bash":
            cmd = ["bash", "-c", command]
            if params:
                cmd.append("--")
                cmd.extend(params)
        elif language == "python":
            cmd = ["python", "-c", command, *params]
        elif language == "c++":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"g++ -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        elif language == "c":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"gcc -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError(f"Unsupported language {language}")
        
        return self.execute_with_retries(cmd)

