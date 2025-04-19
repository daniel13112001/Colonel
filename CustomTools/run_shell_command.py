import subprocess
from typing import Optional
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun
)


class ShellCommandInput(BaseModel):
    command: str = Field(description="The shell command to run")
    args: str = (Field(description="""
    The command line options and positional arguments associated with the command. This is optional.
    """))


#TODO this currently only works for single commands without command line options or flags
class RunShellCommandTool(BaseTool):
    name: str = "RunShellCommandTool"
    description: str = "Useful for when you need to execute a shell command"
    arg_schema: Optional[ArgsSchema] = ShellCommandInput
    return_direct: bool = False  #Do not immediately return to the user after this tool is invoked

    def _run(self,
             command: str,
             args: list = None,
             ) -> str:

        """ Use the tool """

        if args is None:
            args = []

        try:
            result = subprocess.run([command] + args, stdout=True, stderr=True, check=True, text=True).stdout
        except subprocess.CalledProcessError as e:
            result = "Command failed " + e.stderr
        except Exception as e:
            result = "Command failed " + str(e)

        return result

    async def _arun(self,
                    command: str,
                    args: list = None,
                    ) -> str:

        """Use the tool asynchronously"""

        return self._run(command, args)
