import os
import re
from abc import ABC, abstractmethod

from camel.agents import RolePlaying
from camel.messages import ChatMessage
from camel.typing import TaskType, ModelType
from chatdev.chat_env import ChatEnv
from chatdev.statistics import get_info
from chatdev.utils import log_visualize, log_arguments
from chatdev.tools.contract_static import run_contract_intel, ContractIntelError


class Phase(ABC):

    def __init__(self,
                 assistant_role_name,
                 user_role_name,
                 phase_prompt,
                 role_prompts,
                 phase_name,
                 model_type,
                 log_filepath):
        """

        Args:
            assistant_role_name: who receives chat in a phase
            user_role_name: who starts the chat in a phase
            phase_prompt: prompt of this phase
            role_prompts: prompts of all roles
            phase_name: name of this phase
        """
        self.seminar_conclusion = None
        self.assistant_role_name = assistant_role_name
        self.user_role_name = user_role_name
        self.phase_prompt = phase_prompt
        self.phase_env = dict()
        self.phase_name = phase_name
        self.assistant_role_prompt = role_prompts[assistant_role_name]
        self.user_role_prompt = role_prompts[user_role_name]
        self.ceo_prompt = role_prompts["Chief Executive Officer"]
        self.counselor_prompt = role_prompts["Counselor"]
        self.max_retries = 3
        self.reflection_prompt = """Here is a conversation between two roles: {conversations} {question}"""
        self.model_type = model_type
        self.log_filepath = log_filepath

    @log_arguments
    def chatting(
            self,
            chat_env,
            task_prompt: str,
            assistant_role_name: str,
            user_role_name: str,
            phase_prompt: str,
            phase_name: str,
            assistant_role_prompt: str,
            user_role_prompt: str,
            task_type=TaskType.CHATDEV,
            need_reflect=False,
            with_task_specify=False,
            model_type=ModelType.GPT_4_O_MINI,
            memory=None,
            placeholders=None,
            chat_turn_limit=10
    ) -> str:
        """

        Args:
            chat_env: global chatchain environment
            task_prompt: user query prompt for building the software
            assistant_role_name: who receives the chat
            user_role_name: who starts the chat
            phase_prompt: prompt of the phase
            phase_name: name of the phase
            assistant_role_prompt: prompt of assistant role
            user_role_prompt: prompt of user role
            task_type: task type
            need_reflect: flag for checking reflection
            with_task_specify: with task specify
            model_type: model type
            placeholders: placeholders for phase environment to generate phase prompt
            chat_turn_limit: turn limits in each chat

        Returns:

        """

        if placeholders is None:
            placeholders = {}
        assert 1 <= chat_turn_limit <= 100

        if not chat_env.exist_employee(assistant_role_name):
            raise ValueError(f"{assistant_role_name} not recruited in ChatEnv.")
        if not chat_env.exist_employee(user_role_name):
            raise ValueError(f"{user_role_name} not recruited in ChatEnv.")

        # init role play
        role_play_session = RolePlaying(
            assistant_role_name=assistant_role_name,
            user_role_name=user_role_name,
            assistant_role_prompt=assistant_role_prompt,
            user_role_prompt=user_role_prompt,
            task_prompt=task_prompt,
            task_type=task_type,
            with_task_specify=with_task_specify,
            memory=memory,
            model_type=model_type,
            background_prompt=chat_env.config.background_prompt
        )

        # log_visualize("System", role_play_session.assistant_sys_msg)
        # log_visualize("System", role_play_session.user_sys_msg)

        # start the chat
        _, input_user_msg = role_play_session.init_chat(None, placeholders, phase_prompt)
        seminar_conclusion = None

        # handle chats
        # the purpose of the chatting in one phase is to get a seminar conclusion
        # there are two types of conclusion
        # 1. with "<INFO>" mark
        # 1.1 get seminar conclusion flag (ChatAgent.info) from assistant or user role, which means there exist special "<INFO>" mark in the conversation
        # 1.2 add "<INFO>" to the reflected content of the chat (which may be terminated chat without "<INFO>" mark)
        # 2. without "<INFO>" mark, which means the chat is terminated or normally ended without generating a marked conclusion, and there is no need to reflect
        for i in range(chat_turn_limit):
            # start the chat, we represent the user and send msg to assistant
            # 1. so the input_user_msg should be assistant_role_prompt + phase_prompt
            # 2. then input_user_msg send to LLM and get assistant_response
            # 3. now we represent the assistant and send msg to user, so the input_assistant_msg is user_role_prompt + assistant_response
            # 4. then input_assistant_msg send to LLM and get user_response
            # all above are done in role_play_session.step, which contains two interactions with LLM
            # the first interaction is logged in role_play_session.init_chat
            assistant_response, user_response = role_play_session.step(input_user_msg, chat_turn_limit == 1)

            conversation_meta = "**" + assistant_role_name + "<->" + user_role_name + " on : " + str(
                phase_name) + ", turn " + str(i) + "**\n\n"

            # TODO: max_tokens_exceeded errors here
            if isinstance(assistant_response.msg, ChatMessage):
                # we log the second interaction here
                log_visualize(role_play_session.assistant_agent.role_name,
                              conversation_meta + "[" + role_play_session.user_agent.system_message.content + "]\n\n" + assistant_response.msg.content)
                if role_play_session.assistant_agent.info:
                    seminar_conclusion = assistant_response.msg.content
                    break
                if assistant_response.terminated:
                    break

            if isinstance(user_response.msg, ChatMessage):
                # here is the result of the second interaction, which may be used to start the next chat turn
                log_visualize(role_play_session.user_agent.role_name,
                              conversation_meta + "[" + role_play_session.assistant_agent.system_message.content + "]\n\n" + user_response.msg.content)
                if role_play_session.user_agent.info:
                    seminar_conclusion = user_response.msg.content
                    break
                if user_response.terminated:
                    break

            # continue the chat
            if chat_turn_limit > 1 and isinstance(user_response.msg, ChatMessage):
                input_user_msg = user_response.msg
            else:
                break

        # conduct self reflection
        if need_reflect:
            if seminar_conclusion in [None, ""]:
                seminar_conclusion = "<INFO> " + self.self_reflection(task_prompt, role_play_session, phase_name,
                                                                      chat_env)
            if "recruiting" in phase_name:
                if "Yes".lower() not in seminar_conclusion.lower() and "No".lower() not in seminar_conclusion.lower():
                    seminar_conclusion = "<INFO> " + self.self_reflection(task_prompt, role_play_session,
                                                                          phase_name,
                                                                          chat_env)
            elif seminar_conclusion in [None, ""]:
                seminar_conclusion = "<INFO> " + self.self_reflection(task_prompt, role_play_session, phase_name,
                                                                      chat_env)
        else:
            seminar_conclusion = assistant_response.msg.content

        log_visualize("**[Seminar Conclusion]**:\n\n {}".format(seminar_conclusion))
        seminar_conclusion = seminar_conclusion.split("<INFO>")[-1]
        return seminar_conclusion

    def self_reflection(self,
                        task_prompt: str,
                        role_play_session: RolePlaying,
                        phase_name: str,
                        chat_env: ChatEnv) -> str:
        """

        Args:
            task_prompt: user query prompt for building the software
            role_play_session: role play session from the chat phase which needs reflection
            phase_name: name of the chat phase which needs reflection
            chat_env: global chatchain environment

        Returns:
            reflected_content: str, reflected results

        """
        messages = role_play_session.assistant_agent.stored_messages if len(
            role_play_session.assistant_agent.stored_messages) >= len(
            role_play_session.user_agent.stored_messages) else role_play_session.user_agent.stored_messages
        messages = ["{}: {}".format(message.role_name, message.content.replace("\n\n", "\n")) for message in messages]
        messages = "\n\n".join(messages)

        if "recruiting" in phase_name:
            question = """Answer their final discussed conclusion (Yes or No) in the discussion without any other words, e.g., "Yes" """
        elif phase_name == "DemandAnalysis":
            question = """Answer their final product modality in the discussion without any other words, e.g., "PowerPoint" """
        elif phase_name == "LanguageChoose":
            question = """Conclude the programming language being discussed for software development, in the format: "*" where '*' represents a programming language." """
        elif phase_name == "EnvironmentDoc":
            question = """According to the codes and file format listed above, write a requirements.txt file to specify the dependencies or packages required for the project to run properly." """
        elif phase_name == "ContractAnalysis":
            question = """Answer their Solidity smart contract ideas in the discussion without any other words, e.g., "A smart contract where we talk about Security: What functions the contract performs" """
        elif phase_name == "ContractAnalysisCKD":
            question = """Based on the CKE/CKD hybrid detection analysis results, summarize the key security findings, prioritized contracts, and strategic recommendations for the security review process."""
        elif phase_name == "ContractAnalysisET":
            question = """Summarize the final execution trajectory discussed, focusing on fund/state flow and key invariants, without any extra commentary."""
        elif phase_name == "ContractReviewComment":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "BugsSummary":
            question = """Generate analysis reports which locate and summarize the vulnerabilities in contract codes."""
        elif phase_name == "isVul":
            question = """Based on the vulnerability analysis, provide a binary classification: output '1' if the contract is VULNERABLE (contains exploitable vulnerabilities) or '0' if SECURE (no critical exploitable vulnerabilities)."""
        elif phase_name == "TestBugsSummary":
            question = """Perform code audits to identify vulnerabilities and weakness in contract codes."""
        elif phase_name == "ContractModification":
            question = """Modify the contract code based on the error summary."""
        elif phase_name == "ArithmeticDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "ReentrancyDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "UncheckedSendDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        # elif phase_name == "DelegatecallDetector":
        #     question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "TODDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "TimeStampManipulationDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "BadRandDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "TXRelianceDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "SuicideDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "PredictableRandDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "PriceManipulationDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "DataCorruptionDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "WithdrawalFunctionDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "LackAuthorizationDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "DataInconsistencyDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "HashCollisionDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "UninitializedReturnVariableDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "MisdeclaredConstructorDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "MissingOnlyOwnerDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "MisuseMsgValueDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "PrecisionLossDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "RedundantConditionalDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "OracleDependencyDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "OwnershipHijackingDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "CentralizationRiskDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "FundingCalculationDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "FlashLoanDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "MappingGetterDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "GetterFunctionDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "UnnecessaryComparisonDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""     
        elif phase_name == "InconsistentInitializationDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "SourceSwappingDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "SignatureVerificationDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "OrderInitializationDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "ImpracticalityMatchDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "InconsistentTokensDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "PartialWithdrawalsDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "FallbackFunctionDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "UnlimitedTokenDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "InputValidationDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "UncheckedLowLevelCallDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        elif phase_name == "DoSDetector":
            question = """Review and analyze the contract codes, identify the vulnerabilities in codes and and do not discuss anything else."""
        else:
            raise ValueError(f"Reflection of phase {phase_name}: Not Assigned.")

        # Reflections actually is a special phase between CEO and counselor
        # They read the whole chatting history of this phase and give refined conclusion of this phase
        reflected_content = \
            self.chatting(chat_env=chat_env,
                          task_prompt=task_prompt,
                          assistant_role_name="Chief Executive Officer",
                          user_role_name="Counselor",
                          phase_prompt=self.reflection_prompt,
                          phase_name="Reflection",
                          assistant_role_prompt=self.ceo_prompt,
                          user_role_prompt=self.counselor_prompt,
                          placeholders={"conversations": messages, "question": question},
                          need_reflect=False,
                          memory=chat_env.memory,
                          chat_turn_limit=1,
                          model_type=self.model_type)

        if "recruiting" in phase_name:
            if "Yes".lower() in reflected_content.lower():
                return "Yes"
            return "No"
        else:
            return reflected_content

    @abstractmethod
    def update_phase_env(self, chat_env):
        """
        update self.phase_env (if needed) using chat_env, then the chatting will use self.phase_env to follow the context and fill placeholders in phase prompt
        must be implemented in customized phase
        the usual format is just like:
        ```
            self.phase_env.update({key:chat_env[key]})
        ```
        Args:
            chat_env: global chat chain environment

        Returns: None

        """
        pass

    @abstractmethod
    def update_chat_env(self, chat_env) -> ChatEnv:
        """
        update chan_env based on the results of self.execute, which is self.seminar_conclusion
        must be implemented in customized phase
        the usual format is just like:
        ```
            chat_env.xxx = some_func_for_postprocess(self.seminar_conclusion)
        ```
        Args:
            chat_env:global chat chain environment

        Returns:
            chat_env: updated global chat chain environment

        """
        pass

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        """
        execute the chatting in this phase
        1. receive information from environment: update the phase environment from global environment
        2. execute the chatting
        3. change the environment: update the global environment using the conclusion
        Args:
            chat_env: global chat chain environment
            chat_turn_limit: turn limit in each chat
            need_reflect: flag for reflection

        Returns:
            chat_env: updated global chat chain environment using the conclusion from this phase execution

        """
        self.update_phase_env(chat_env)
        self.seminar_conclusion = \
            self.chatting(chat_env=chat_env,
                          task_prompt=chat_env.env_dict['task_prompt'],
                          need_reflect=need_reflect,
                          assistant_role_name=self.assistant_role_name,
                          user_role_name=self.user_role_name,
                          phase_prompt=self.phase_prompt,
                          phase_name=self.phase_name,
                          assistant_role_prompt=self.assistant_role_prompt,
                          user_role_prompt=self.user_role_prompt,
                          chat_turn_limit=chat_turn_limit,
                          placeholders=self.phase_env,
                          memory=chat_env.memory,
                          model_type=self.model_type)
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class DemandAnalysis(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        pass

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0:
            chat_env.env_dict['modality'] = self.seminar_conclusion.split("<INFO>")[-1].lower().replace(".", "").strip()
        return chat_env


class LanguageChoose(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "description": chat_env.env_dict['task_description'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['language'] = self.seminar_conclusion.split("<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['language'] = self.seminar_conclusion
        else:
            chat_env.env_dict['language'] = "Python"
        return chat_env


class Coding(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        gui = "" if not chat_env.config.gui_design \
            else "The software should be equipped with graphical user interface (GUI) so that user can visually and graphically use it; so you must choose a GUI framework (e.g., in Python, you can implement GUI via tkinter, Pygame, Flexx, PyGUI, etc,)."
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "description": chat_env.env_dict['task_description'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "gui": gui})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.update_codes(self.seminar_conclusion)
        if len(chat_env.codes.codebooks.keys()) == 0:
            raise ValueError("No Valid Codes.")
        chat_env.rewrite_codes("Finish Coding")
        log_visualize(
            "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class ArtDesign(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env = {"task": chat_env.env_dict['task_prompt'],
                          "description": chat_env.env_dict['task_description'],
                          "language": chat_env.env_dict['language'],
                          "codes": chat_env.get_codes()}

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.proposed_images = chat_env.get_proposed_images_from_message(self.seminar_conclusion)
        log_visualize(
            "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class ArtIntegration(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env = {"task": chat_env.env_dict['task_prompt'],
                          "language": chat_env.env_dict['language'],
                          "codes": chat_env.get_codes(),
                          "images": "\n".join(
                              ["{}: {}".format(filename, chat_env.proposed_images[filename]) for
                               filename in sorted(list(chat_env.proposed_images.keys()))])}

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.update_codes(self.seminar_conclusion)
        chat_env.rewrite_codes("Finish Art Integration")
        # chat_env.generate_images_from_codes()
        log_visualize(
            "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class CodeComplete(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "unimplemented_file": ""})
        unimplemented_file = ""
        for filename in self.phase_env['pyfiles']:
            code_content = open(os.path.join(chat_env.env_dict['directory'], filename)).read()
            lines = [line.strip() for line in code_content.split("\n") if line.strip() == "pass"]
            if len(lines) > 0 and self.phase_env['num_tried'][filename] < self.phase_env['max_num_implement']:
                unimplemented_file = filename
                break
        self.phase_env['num_tried'][unimplemented_file] += 1
        self.phase_env['unimplemented_file'] = unimplemented_file

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.update_codes(self.seminar_conclusion)
        if len(chat_env.codes.codebooks.keys()) == 0:
            raise ValueError("No Valid Codes.")
        chat_env.rewrite_codes("Code Complete #" + str(self.phase_env["cycle_index"]) + " Finished")
        log_visualize(
            "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class CodeReviewComment(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "modality": chat_env.env_dict['modality'],
             "ideas": chat_env.env_dict['ideas'],
             "language": chat_env.env_dict['language'],
             "codes": chat_env.get_codes(),
             "images": ", ".join(chat_env.incorporated_images)})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['review_comments'] = self.seminar_conclusion
        return chat_env


class CodeReviewModification(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "comments": chat_env.env_dict['review_comments']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if "```".lower() in self.seminar_conclusion.lower():
            chat_env.update_codes(self.seminar_conclusion)
            chat_env.rewrite_codes("Review #" + str(self.phase_env["cycle_index"]) + " Finished")
            log_visualize(
                "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env


class CodeReviewHuman(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes()})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if "```".lower() in self.seminar_conclusion.lower():
            chat_env.update_codes(self.seminar_conclusion)
            chat_env.rewrite_codes("Human Review #" + str(self.phase_env["cycle_index"]) + " Finished")
            log_visualize(
                "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        log_visualize(
            f"**[Human-Agent-Interaction]**\n\n"
            f"Now you can participate in the development of the software!\n"
            f"The task is:  {chat_env.env_dict['task_prompt']}\n"
            f"Please input your feedback (in multiple lines). It can be bug report or new feature requirement.\n"
            f"You are currently in the #{self.phase_env['cycle_index']} human feedback with a total of {self.phase_env['cycle_num']} feedbacks\n"
            f"Type 'end' on a separate line to submit.\n"
            f"You can type \"Exit\" to quit this mode at any time.\n"
        )
        provided_comments = []
        while True:
            user_input = input(">>>>>>")
            if user_input.strip().lower() == "end":
                break
            if user_input.strip().lower() == "exit":
                provided_comments = ["exit"]
                break
            provided_comments.append(user_input)
        self.phase_env["comments"] = '\n'.join(provided_comments)
        log_visualize(
            f"**[User Provided Comments]**\n\n In the #{self.phase_env['cycle_index']} of total {self.phase_env['cycle_num']} comments: \n\n" +
            self.phase_env["comments"])
        if self.phase_env["comments"].strip().lower() == "exit":
            return chat_env

        self.seminar_conclusion = \
            self.chatting(chat_env=chat_env,
                          task_prompt=chat_env.env_dict['task_prompt'],
                          need_reflect=need_reflect,
                          assistant_role_name=self.assistant_role_name,
                          user_role_name=self.user_role_name,
                          phase_prompt=self.phase_prompt,
                          phase_name=self.phase_name,
                          assistant_role_prompt=self.assistant_role_prompt,
                          user_role_prompt=self.user_role_prompt,
                          chat_turn_limit=chat_turn_limit,
                          placeholders=self.phase_env,
                          memory=chat_env.memory,
                          model_type=self.model_type)
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class TestErrorSummary(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        chat_env.generate_images_from_codes()
        (exist_bugs_flag, test_reports) = chat_env.exist_bugs()
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "test_reports": test_reports,
                               "exist_bugs_flag": exist_bugs_flag})
        log_visualize("**[Test Reports]**:\n\n{}".format(test_reports))

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['error_summary'] = self.seminar_conclusion
        chat_env.env_dict['test_reports'] = self.phase_env['test_reports']

        return chat_env

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        if "ModuleNotFoundError" in self.phase_env['test_reports']:
            chat_env.fix_module_not_found_error(self.phase_env['test_reports'])
            log_visualize(
                f"Software Test Engineer found ModuleNotFoundError:\n{self.phase_env['test_reports']}\n")
            pip_install_content = ""
            for match in re.finditer(r"No module named '(\S+)'", self.phase_env['test_reports'], re.DOTALL):
                module = match.group(1)
                pip_install_content += "{}\n```{}\n{}\n```\n".format("cmd", "bash", f"pip install {module}")
                log_visualize(f"Programmer resolve ModuleNotFoundError by:\n{pip_install_content}\n")
            self.seminar_conclusion = "nothing need to do"
        else:
            self.seminar_conclusion = \
                self.chatting(chat_env=chat_env,
                              task_prompt=chat_env.env_dict['task_prompt'],
                              need_reflect=need_reflect,
                              assistant_role_name=self.assistant_role_name,
                              user_role_name=self.user_role_name,
                              phase_prompt=self.phase_prompt,
                              phase_name=self.phase_name,
                              assistant_role_prompt=self.assistant_role_prompt,
                              user_role_prompt=self.user_role_prompt,
                              memory=chat_env.memory,
                              chat_turn_limit=chat_turn_limit,
                              placeholders=self.phase_env)
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class TestModification(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "test_reports": chat_env.env_dict['test_reports'],
                               "error_summary": chat_env.env_dict['error_summary'],
                               "codes": chat_env.get_codes()
                               })

    def update_chat_env(self, chat_env) -> ChatEnv:
        if "```".lower() in self.seminar_conclusion.lower():
            chat_env.update_codes(self.seminar_conclusion)
            chat_env.rewrite_codes("Test #" + str(self.phase_env["cycle_index"]) + " Finished")
            log_visualize(
                "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class EnvironmentDoc(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes()})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env._update_requirements(self.seminar_conclusion)
        chat_env.rewrite_requirements()
        log_visualize(
            "**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'], self.log_filepath)))
        return chat_env


class Manual(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "requirements": chat_env.get_requirements()})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env._update_manuals(self.seminar_conclusion)
        chat_env.rewrite_manuals()
        return chat_env
    
class ContractAnalysis(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env = {"task": chat_env.env_dict['task_prompt']}

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class ContractAnalysisCKD(Phase):
    """
    基于CKE/CKD混合检测的合约分析Phase
    三阶段流程: 静态规则筛选 → 风险画像评分 → LLM精细分析
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.budget = kwargs.get('budget', 'medium')  # low/medium/high
    
    def update_phase_env(self, chat_env):
        """运行混合检测流程并准备环境变量"""
        base_dir = chat_env.env_dict.get('directory', '')
        hint = chat_env.env_dict.get('solidity_file', '')
        task_prompt = chat_env.env_dict.get('task_prompt', '')
        
        # 如果task_prompt是.sol文件路径，优先使用它
        if task_prompt.endswith('.sol') and os.path.isfile(task_prompt):
            sol_file = os.path.abspath(task_prompt)
            log_visualize(f"**[ContractAnalysisCKD]** Using task_prompt as solidity file: {sol_file}")
        else:
            try:
                sol_file = self._discover_solidity_file(base_dir, hint)
            except FileNotFoundError:
                sol_file = hint or task_prompt or "<unknown>"
                log_visualize(f"**[ContractAnalysisCKD]** Solidity file not found in {base_dir}")
                self.phase_env = {
                    "task": task_prompt,
                    "analysis_status": "file_not_found",
                    "budget_level": self.budget,
                    "total_contracts": 0,
                    "filtered_contracts": 0,
                    "analyzed_contracts": 0,
                    "risk_summary": "No Solidity file found for analysis.",
                    "contract_details": ""
                }
                return
        
        # 自动检测并切换 Solidity 版本（模仿 ContractAnalysisET）
        try:
            from chatdev.tools.contract_static import (
                _detect_solidity_version,
                _get_available_solc_versions,
                _select_best_solc_version,
                _switch_solc_version
            )
            from pathlib import Path
            
            sol_path = Path(sol_file)
            full_ver, major_minor, prefix = _detect_solidity_version(sol_path)
            available = _get_available_solc_versions()
            
            if available:
                best_version = _select_best_solc_version(full_ver, major_minor, prefix, available)
                if best_version:
                    _switch_solc_version(best_version)
                    log_visualize(f"**[ContractAnalysisCKD]** 🔧 Using Solidity {best_version} for {sol_path.name}")
                else:
                    log_visualize(f"**[ContractAnalysisCKD][Warning]** No compatible solc version for {prefix or ''}{full_ver}")
            else:
                log_visualize(f"**[ContractAnalysisCKD][Warning]** solc-select not available, using default version")
                
        except Exception as version_exc:
            log_visualize(f"**[ContractAnalysisCKD][Warning]** Version auto-selection failed: {version_exc}")
        
        # 运行混合检测流程
        from scripts.hybrid_detection_pipeline import HybridDetectionPipeline
        
        try:
            log_visualize(f"**[ContractAnalysisCKD]** Starting hybrid detection: {sol_file} (budget={self.budget})")
            pipeline = HybridDetectionPipeline(sol_file, budget=self.budget)
            results = pipeline.run()
            
            # 提取关键信息
            risk_profiles = results.get('profiles', [])
            analyzed_contracts = results.get('results', [])
            
            # 保存原始数据供VulnerabilityProbing使用
            self._analyzed_contracts_raw = analyzed_contracts
            self._risk_profiles_raw = risk_profiles  # 保存风险画像数据
            
            # 检查是否禁用了CKD（路径蒸馏）
            is_ckd_disabled = not pipeline.config.get('path_distillation', True)
            if is_ckd_disabled:
                log_visualize(f"**[ContractAnalysisCKD]** ⚠️ CKD (路径蒸馏) 已禁用，使用简化分析")
            
            # 检查是否跳过了阶段3（C_no_ckd_cke消融实验）
            is_stage3_skipped = pipeline.config.get('skip_stage3', False)
            if is_stage3_skipped:
                log_visualize(f"**[ContractAnalysisCKD]** ⚠️ 阶段3已跳过，将使用阶段2风险画像进行检测")
                self._skip_stage3 = True
            
            # 构建LLM可读的风险摘要（静态方法，不传递ckd_disabled参数）
            risk_summary = self._format_risk_summary(risk_profiles)
            contract_details = self._format_contract_details(analyzed_contracts)
            
            log_visualize(
                f"**[ContractAnalysisCKD]** Hybrid detection completed\n"
                f"  - Total contracts: {results['total_contracts']}\n"
                f"  - After filtering: {results['filtered_contracts']}\n"
                f"  - Analyzed (budget={self.budget}): {results['analyzed_contracts']}\n"
                f"  - Token savings: ~{100 - results['analyzed_contracts']*10}%"
            )
            
            self.phase_env = {
                "task": chat_env.env_dict['task_prompt'],
                "solidity_file": sol_file,
                "analysis_status": "success",
                "budget_level": self.budget,
                "total_contracts": results['total_contracts'],
                "filtered_contracts": results['filtered_contracts'],
                "analyzed_contracts": results['analyzed_contracts'],
                "risk_summary": risk_summary,
                "contract_details": contract_details
            }
            
        except Exception as exc:
            log_visualize(f"**[ContractAnalysisCKD][Error]** {exc}")
            self.phase_env = {
                "task": chat_env.env_dict['task_prompt'],
                "analysis_status": f"error: {exc}",
                "budget_level": self.budget,
                "total_contracts": 0,
                "filtered_contracts": 0,
                "analyzed_contracts": 0,
                "risk_summary": f"Analysis failed: {exc}",
                "contract_details": ""
            }
    
    def update_chat_env(self, chat_env) -> ChatEnv:
        """更新全局环境"""
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "Hybrid CKD detection analysis completed"
        
        # 保存检测结果供后续Phase使用
        chat_env.env_dict['ckd_analysis'] = {
            'risk_summary': self.phase_env.get('risk_summary', ''),
            'contract_details': self.phase_env.get('contract_details', ''),
            'budget': self.budget,
            'analyzed_contracts': self.phase_env.get('analyzed_contracts', 0)
        }
        
        # 保存原始analyzed_contracts数据供VulnerabilityProbing使用
        if hasattr(self, '_analyzed_contracts_raw'):
            chat_env.env_dict['_analyzed_contracts_raw'] = self._analyzed_contracts_raw
        
        # 保存风险画像数据供VulnerabilityProbing使用（用于C_no_ckd_cke模式）
        if hasattr(self, '_risk_profiles_raw'):
            chat_env.env_dict['_risk_profiles_raw'] = self._risk_profiles_raw
        
        # 传递skip_stage3标志
        if hasattr(self, '_skip_stage3'):
            chat_env.env_dict['_skip_stage3'] = self._skip_stage3
        
        return chat_env
    
    @staticmethod
    def _discover_solidity_file(base_dir: str, hint: str = "") -> str:
        """发现Solidity文件（复用ContractAnalysisET的逻辑）"""
        def _candidate(path: str) -> str | None:
            if path and os.path.isfile(path) and path.lower().endswith('.sol'):
                return path
            return None

        if hint:
            expanded = os.path.abspath(os.path.join(base_dir, hint)) if not os.path.isabs(hint) else hint
            candidate = _candidate(expanded)
            if candidate:
                return candidate

        if base_dir and os.path.isdir(base_dir):
            for root, _, files in os.walk(base_dir):
                for filename in files:
                    if filename.lower().endswith('.sol'):
                        return os.path.join(root, filename)

        raise FileNotFoundError("No Solidity (.sol) file found for CKD analysis.")
    
    @staticmethod
    def _format_risk_summary(profiles: list) -> str:
        """格式化风险摘要供LLM理解"""
        if not profiles:
            return "No contracts passed the initial filtering stage (all were interfaces, libraries, or read-only contracts)."
        
        summary_lines = []
        for i, profile in enumerate(profiles[:5], 1):  # Top-5
            risk_score = profile.get('risk_score', 0)
            name = profile.get('name', 'Unknown')
            complexity = profile.get('complexity_score', 0)
            sensitive_ops = profile.get('sensitive_operations', [])
            indicators = profile.get('vulnerability_indicators', [])
            
            risk_level = "🔴 HIGH RISK" if risk_score >= 20 else "🟡 MEDIUM RISK" if risk_score >= 10 else "🟢 LOW RISK"
            
            summary_lines.append(f"**{i}. Contract `{name}`** - {risk_level} (Score: {risk_score:.1f}, Complexity: {complexity})")
            
            if sensitive_ops:
                summary_lines.append(f"   - Sensitive Operations: {', '.join(sensitive_ops[:3])}")
            
            if indicators:
                summary_lines.append(f"   - Vulnerability Indicators:")
                for indicator in indicators[:3]:
                    summary_lines.append(f"     • {indicator}")
            
            summary_lines.append("")
        
        return "\n".join(summary_lines) if summary_lines else "No significant risks identified."
    
    @staticmethod
    def _format_contract_details(analyzed: list) -> str:
        """格式化合约详情供LLM分析"""
        if not analyzed:
            return "No contracts selected for detailed analysis based on budget constraints."
        
        detail_lines = []
        for result in analyzed:
            contract = result.get('contract', 'Unknown')
            risk_score = result.get('risk_score', 0)
            recommendation = result.get('recommendation', '')
            sensitive_ops = result.get('sensitive_operations', [])
            unprotected_funcs = result.get('unprotected_functions', [])
            indicators = result.get('vulnerability_indicators', [])
            ckd_analysis = result.get('ckd_analysis', {})
            
            detail_lines.append(f"### Contract: `{contract}`")
            detail_lines.append(f"**Risk Score**: {risk_score:.1f}")
            detail_lines.append(f"**Recommendation**: {recommendation}\n")
            
            if sensitive_ops:
                detail_lines.append(f"**Sensitive Operations Detected**:")
                for op in sensitive_ops:
                    detail_lines.append(f"- {op}")
                detail_lines.append("")
            
            if unprotected_funcs:
                detail_lines.append(f"**Functions Lacking Access Control** ({len(unprotected_funcs)}):")
                for func in unprotected_funcs[:5]:  # 最多显示5个
                    detail_lines.append(f"- `{func}()`")
                if len(unprotected_funcs) > 5:
                    detail_lines.append(f"- ... and {len(unprotected_funcs) - 5} more")
                detail_lines.append("")
            
            if indicators:
                detail_lines.append(f"**Vulnerability Indicators**:")
                for indicator in indicators:
                    detail_lines.append(f"- {indicator}")
                detail_lines.append("")
            
            # 添加CKD深度分析结果（如果有）
            # 注意：当path_distillation=False时，ckd_analysis可能为空或只有error信息
            if ckd_analysis and isinstance(ckd_analysis, dict) and 'path_details' in ckd_analysis:
                path_details = ckd_analysis.get('path_details', [])
                if path_details and len(path_details) > 0:
                    detail_lines.append(f"**CKD Deep Analysis** ({ckd_analysis.get('total_paths', 0)} paths, {ckd_analysis.get('high_risk_functions', 0)} functions):")
                    detail_lines.append("")
                    
                    for i, path in enumerate(path_details, 1):
                        detail_lines.append(f"**Path {i}: `{path.get('function', 'Unknown')}()` → {path.get('sink_type', 'unknown')}**")
                        detail_lines.append(f"  - Risk Score: {path.get('risk_score', 0):.1f}")
                        
                        risk_factors = path.get('risk_factors', [])
                        if risk_factors:
                            detail_lines.append(f"  - Risk Factors:")
                            for factor in risk_factors:
                                detail_lines.append(f"    • {factor}")
                        
                        guards = path.get('guards', [])
                        if guards:
                            detail_lines.append(f"  - Guard Conditions: {len(guards)}")
                            for guard in guards[:2]:  # 显示前2个守卫
                                detail_lines.append(f"    • `{guard}`")
                        else:
                            detail_lines.append(f"  - ⚠️ No guard conditions found")
                        
                        state_writes = path.get('state_writes', [])
                        if state_writes:
                            detail_lines.append(f"  - State Modifications: {', '.join(state_writes)}")
                        
                        detail_lines.append("")
            elif ckd_analysis and 'error' in ckd_analysis:
                detail_lines.append(f"**CKD Analysis**: Failed - {ckd_analysis['error']}")
                detail_lines.append("")
        
        return "\n".join(detail_lines) if detail_lines else "No detailed analysis results available."


class VulnerabilityProbing(Phase):
    """
    基于CKD路径结果的漏洞探测Phase
    按函数逐个构造prompt_builder风格的提示词，询问LLM是否存在漏洞
    一旦发现漏洞就终止，否则遍历所有函数
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.function_contexts = []  # 存储按函数分组的路径上下文
        self.current_function_index = 0
        self.vulnerability_found = False
        self.vulnerability_details = None
    
    def update_phase_env(self, chat_env):
        """从上一阶段提取路径数据或风险画像，按函数分组"""
        # 检查是否跳过了阶段3（C_no_ckd_cke消融实验）
        skip_stage3 = chat_env.env_dict.get('_skip_stage3', False)
        
        if skip_stage3:
            # 使用阶段2的风险画像数据进行检测
            log_visualize("**[VulnerabilityProbing]** 🔄 使用阶段2风险画像进行漏洞探测（C_no_ckd_cke模式）")
            risk_profiles = chat_env.env_dict.get('_risk_profiles_raw', [])
            
            if not risk_profiles:
                log_visualize("**[VulnerabilityProbing]** 没有风险画像数据，跳过探测")
                self.phase_env = {
                    "probing_status": "no_data",
                    "message": "No risk profile data available"
                }
                chat_env.env_dict['_llm_probing_executed'] = False
                return
            
            # 将风险画像转换为函数级探测任务
            self._prepare_risk_based_probing(risk_profiles)
            return
        
        # 正常模式：使用CKD路径数据
        analyzed_contracts = chat_env.env_dict.get('_analyzed_contracts_raw', [])
        
        if not analyzed_contracts:
            log_visualize("**[VulnerabilityProbing]** 没有检测结果，跳过探测")
            self.phase_env = {
                "probing_status": "no_data",
                "message": "No CKD analysis results to probe"
            }
            # 标记为未进行LLM探测
            chat_env.env_dict['_llm_probing_executed'] = False
            return
        
        # 按函数分组路径
        for contract_result in analyzed_contracts:
            contract_name = contract_result.get('contract', 'Unknown')
            ckd_data = contract_result.get('ckd_analysis', {})
            path_details = ckd_data.get('path_details', [])
            
            if not path_details:
                log_visualize(f"**[VulnerabilityProbing]** 合约 {contract_name} 无路径数据，跳过")
                continue
            
            # 按函数分组
            func_groups = {}
            for path in path_details:
                func_name = path.get('function', 'Unknown')
                if func_name not in func_groups:
                    func_groups[func_name] = []
                func_groups[func_name].append(path)
            
            # 为每个函数创建上下文
            for func_name, paths in func_groups.items():
                context = {
                    'contract_name': contract_name,
                    'function_signature': f"{func_name}()",
                    'paths': paths,
                    'total_risk_score': sum(p.get('risk_score', 0) for p in paths)
                }
                self.function_contexts.append(context)
        
        log_visualize(f"**[VulnerabilityProbing]** 准备探测 {len(self.function_contexts)} 个函数")
        
        # 准备第一个函数的prompt
        if self.function_contexts:
            self._prepare_current_function_env()
        else:
            self.phase_env = {
                "probing_status": "no_functions",
                "message": "No functions with paths to probe"
            }
    
    def _prepare_risk_based_probing(self, risk_profiles):
        """基于阶段2风险画像准备探测任务（C_no_ckd_cke模式）"""
        log_visualize(f"**[VulnerabilityProbing]** 基于 {len(risk_profiles)} 个合约的风险画像进行探测")
        
        # 转换风险画像为探测上下文
        # 在C_no_ckd_cke模式下，即使风险评分为0也进行推断性探测
        for profile in risk_profiles:
            # 提取正确的字段名
            complexity = profile.get('complexity_score', 0)
            sensitive_ops = profile.get('sensitive_operations', [])
            unprotected_funcs = profile.get('unprotected_functions', [])
            
            context = {
                'contract_name': profile.get('name', 'Unknown'),
                'risk_score': profile.get('risk_score', 0),
                'complexity': complexity,
                'sensitive_ops': len(sensitive_ops) if isinstance(sensitive_ops, list) else sensitive_ops,
                'unprotected_functions': len(unprotected_funcs) if isinstance(unprotected_funcs, list) else unprotected_funcs,
                'mode': 'risk_profile'  # 标记为风险画像模式
            }
            self.function_contexts.append(context)
        
        if not self.function_contexts:
            log_visualize(f"**[VulnerabilityProbing]** 没有合约可探测")
            self.phase_env = {
                "probing_status": "no_data",
                "message": "No contracts available for probing"
            }
            return
        
        # 按风险评分降序排序
        self.function_contexts.sort(key=lambda x: x.get('risk_score', 0), reverse=True)
        
        log_visualize(f"**[VulnerabilityProbing]** 准备探测 {len(self.function_contexts)} 个合约（包括低风险合约）")
        
        # 准备第一个合约的prompt
        self._prepare_risk_profile_env()
    
    def _prepare_risk_profile_env(self):
        """准备基于风险画像的环境变量（C_no_ckd_cke模式）"""
        if self.current_function_index >= len(self.function_contexts):
            self.phase_env = {
                "probing_status": "completed",
                "total_contracts": len(self.function_contexts),
                "message": "基于风险画像的探测已完成"
            }
            return
        
        ctx = self.function_contexts[self.current_function_index]
        
        self.phase_env = {
            "probing_status": "risk_probing",
            "current_index": self.current_function_index + 1,
            "total_contracts": len(self.function_contexts),
            "contract_name": ctx['contract_name'],
            "risk_score": f"{ctx['risk_score']:.1f}",
            "complexity": ctx['complexity'],
            "sensitive_ops": ctx['sensitive_ops'],
            "unprotected_functions": ctx['unprotected_functions'],
            "mode": "risk_profile"
        }
    
    def _prepare_current_function_env(self):
        """准备当前函数的环境变量（prompt_builder格式）"""
        if self.current_function_index >= len(self.function_contexts):
            self.phase_env = {
                "probing_status": "completed",
                "total_functions": len(self.function_contexts),
                "message": "所有函数均未发现可利用漏洞"
            }
            return
        
        ctx = self.function_contexts[self.current_function_index]
        
        # 获取第一个路径的详细信息（所有路径共享相同的函数和状态变量定义）
        first_path = ctx['paths'][0] if ctx['paths'] else {}
        
        # 使用完整的函数签名（如果有）
        function_signature = first_path.get('function_signature', ctx['function_signature'])
        visibility = first_path.get('visibility', 'public')
        modifiers = first_path.get('modifiers', [])
        
        # 构造状态变量部分（带类型定义）
        state_var_defs = first_path.get('state_var_definitions', {})
        all_state_vars = set()
        for path in ctx['paths']:
            all_state_vars.update(path.get('state_writes', []))
            all_state_vars.update(path.get('state_reads', []))
        
        state_variables_section = "### 相关状态变量\n```solidity\n"
        if all_state_vars and state_var_defs:
            for var in sorted(all_state_vars):
                var_type = state_var_defs.get(var, 'unknown')
                state_variables_section += f"{var_type} {var};\n"
        elif all_state_vars:
            # 回退：如果没有类型定义，只显示变量名
            state_variables_section += "// State variables (types unavailable):\n"
            for var in sorted(all_state_vars):
                state_variables_section += f"// {var}\n"
        else:
            state_variables_section += "// No state variables accessed\n"
        state_variables_section += "```"
        
        # 构造依赖函数部分
        dependent_funcs = first_path.get('dependent_functions', {})
        dependent_functions_section = ""
        if dependent_funcs:
            dependent_functions_section = "\n### 依赖函数\n```solidity\n"
            for func_name, func_code in dependent_funcs.items():
                dependent_functions_section += f"{func_code}\n\n"
            dependent_functions_section += "```"
        
        # 构造路径切片部分
        path_slices_section = self._build_paths_section(ctx['paths'])
        
        self.phase_env = {
            "probing_status": "probing",
            "current_index": self.current_function_index + 1,
            "total_functions": len(self.function_contexts),
            "contract_name": ctx['contract_name'],
            "function_signature": function_signature,
            "visibility": visibility,
            "modifiers": ', '.join(modifiers) if modifiers else '无',
            "total_risk_score": f"{ctx['total_risk_score']:.1f}",
            "state_variables_section": state_variables_section,
            "dependent_functions_section": dependent_functions_section,
            "path_slices_section": path_slices_section
        }
    
    def _build_paths_section(self, paths):
        """构造路径切片部分（prompt_builder格式）"""
        lines = []
        
        for i, path in enumerate(paths, 1):
            slice_id = f"{path.get('function', 'func')}_sink{i-1}"
            lines.append(f"### 路径 {i}: `{slice_id}`")
            lines.append(f"**风险评分**: {path.get('risk_score', 0):.1f}")
            lines.append(f"**Sink 类型**: {path.get('sink_type', 'unknown')}")
            
            # 风险因素
            risk_factors = path.get('risk_factors', [])
            if risk_factors:
                lines.append("**风险因素**:")
                for factor in risk_factors:
                    lines.append(f"- {factor}")
            
            # 守卫条件
            guards = path.get('guards', [])
            if guards:
                lines.append("")
                lines.append("**守卫条件** (必须满足才能到达 Sink):")
                lines.append("```solidity")
                for guard in guards:
                    lines.append(f"  {guard}")
                lines.append("```")
            else:
                lines.append("")
                lines.append("⚠️ **该路径没有守卫条件！**")
            
            # 状态变量访问
            state_reads = path.get('state_reads', [])
            state_writes = path.get('state_writes', [])
            if state_reads or state_writes:
                lines.append("")
                lines.append("**状态变量访问**:")
                if state_reads:
                    lines.append(f"- 读取: {', '.join(sorted(state_reads))}")
                if state_writes:
                    lines.append(f"- 写入: {', '.join(sorted(state_writes))}")
            
            # 依赖函数调用
            dependent_funcs = path.get('dependent_function_list', [])
            if dependent_funcs:
                lines.append("")
                lines.append("**调用的其他函数**:")
                for func in dependent_funcs:
                    lines.append(f"- `{func}()`")
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _load_risk_profile_phase_prompt(self):
        """返回基于风险画像的提示词（直接返回，不从配置文件读取）"""
        return self._build_risk_profile_prompt()
    
    def _build_path_based_prompt(self):
        """构建基于CKD路径的提示词（原版）"""
        # 读取PhaseConfig中的原始提示词模板
        # 这里直接返回空，让chatting方法使用配置文件中的phase_prompt
        return None
    
    def _build_risk_profile_prompt(self):
        """构建基于风险画像的提示词（C_no_ckd_cke模式 - 简化版）"""
        prompt_lines = [
            "你是一位智能合约安全分析专家，负责进行智能合约漏洞的二分类判断。",
            "",
            "## 核心任务",
            "请仔细审查以下智能合约的**源代码**，基于代码逻辑判断该合约是否存在安全漏洞。",
            "",
            "## 合约信息",
            "",
            "**合约名称**: `{contract_name}`",
            "**风险评分**: {risk_score}（参考值）",
            "",
            "## 输出格式要求",
            "",
            "严格按照以下格式输出：",
            "",
            "```",
            "代码审查: [基于源代码的具体分析，引用关键代码行，不超过100字]",
            "漏洞存在: [是/否]",
            "```"
        ]
        
        return "\n".join(prompt_lines)
    
    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        """重写execute方法，实现循环探测逻辑"""
        self.update_phase_env(chat_env)
        
        # 如果没有数据或已完成，直接返回
        probing_status = self.phase_env.get('probing_status')
        if probing_status in ['no_data', 'no_functions', 'completed', 'low_risk']:
            log_visualize(f"**[VulnerabilityProbing]** {self.phase_env.get('message', 'Nothing to probe')}")
            return self.update_chat_env(chat_env)
        
        # 标记为开始执行LLM探测
        chat_env.env_dict['_llm_probing_executed'] = True
        
        # 根据模式选择提示词
        is_risk_mode = (probing_status == 'risk_probing')
        
        # 循环探测每个函数或合约
        while self.current_function_index < len(self.function_contexts) and not self.vulnerability_found:
            ctx = self.function_contexts[self.current_function_index]
            
            # 动态生成phase_prompt
            if is_risk_mode:
                phase_prompt = self._build_risk_profile_prompt()
            else:
                phase_prompt = self._build_path_based_prompt()
            
            # 执行单次chatting
            try:
                if is_risk_mode:
                    log_visualize(
                        f"**[VulnerabilityProbing]** 探测合约 {self.current_function_index + 1}/{len(self.function_contexts)}: "
                        f"{self.phase_env['contract_name']} (风险评分: {self.phase_env['risk_score']})"
                    )
                else:
                    log_visualize(
                        f"**[VulnerabilityProbing]** 探测函数 {self.current_function_index + 1}/{len(self.function_contexts)}: "
                        f"{self.phase_env['contract_name']}.{self.phase_env['function_signature']}"
                    )
                
                # 选择提示词
                if is_risk_mode:
                    # 风险画像模式：从PhaseConfig.json加载VulnerabilityProbingRiskProfile提示词
                    selected_phase_prompt = self._load_risk_profile_phase_prompt()
                else:
                    # CKD路径模式：使用动态构建的提示词
                    selected_phase_prompt = phase_prompt if phase_prompt else self.phase_prompt
                
                response = self.chatting(
                    chat_env=chat_env,
                    task_prompt=chat_env.env_dict['task_prompt'],
                    need_reflect=False,
                    assistant_role_name=self.assistant_role_name,
                    user_role_name=self.user_role_name,
                    phase_prompt=selected_phase_prompt,
                    phase_name=self.phase_name,  # 始终使用VulnerabilityProbing
                    assistant_role_prompt=self.assistant_role_prompt,
                    user_role_prompt=self.user_role_prompt,
                    chat_turn_limit=1,  # 每次探测只需要一轮对话
                    placeholders=self.phase_env,
                    memory=chat_env.memory,
                    model_type=self.model_type
                )
                
                # 解析LLM响应
                has_vulnerability = self._parse_vulnerability_response(response)
                
                if has_vulnerability:
                    self.vulnerability_found = True
                    func_or_contract = self.phase_env.get('function_signature', self.phase_env['contract_name'])
                    self.vulnerability_details = {
                        'contract': self.phase_env['contract_name'],
                        'function': func_or_contract,
                        'response': response,
                        'index': self.current_function_index + 1
                    }
                    log_visualize(f"**[VulnerabilityProbing]** ⚠️ 发现漏洞！{'合约' if is_risk_mode else '函数'}: {func_or_contract}")
                    break
                else:
                    func_or_contract = self.phase_env.get('function_signature', self.phase_env['contract_name'])
                    log_visualize(f"**[VulnerabilityProbing]** ✅ {'合约' if is_risk_mode else '函数'}安全: {func_or_contract}")
                    self.current_function_index += 1
                    # 准备下一个函数或合约
                    if self.current_function_index < len(self.function_contexts):
                        if is_risk_mode:
                            self._prepare_risk_profile_env()
                        else:
                            self._prepare_current_function_env()
                
            except Exception as e:
                log_visualize(f"**[VulnerabilityProbing][Error]** {e}")
                import traceback
                traceback.print_exc()
                # 标记为执行失败
                chat_env.env_dict['_llm_probing_executed'] = False
                break
        
        # 更新环境
        return self.update_chat_env(chat_env)
    
    def _parse_vulnerability_response(self, response: str) -> bool:
        """解析LLM响应，判断是否发现漏洞"""
        response_lower = response.lower()
        
        # 匹配"漏洞存在: 是"
        if '漏洞存在' in response:
            lines = response.split('\n')
            for line in lines:
                if '漏洞存在' in line:
                    if '是' in line and '否' not in line:
                        return True
                    # 处理"漏洞存在: 是"格式
                    if ':' in line or '：' in line:
                        parts = line.split(':') if ':' in line else line.split('：')
                        if len(parts) > 1 and '是' in parts[1].strip():
                            return True
        
        # 匹配"漏洞类型: " 且不是"无"
        if '漏洞类型' in response:
            lines = response.split('\n')
            for line in lines:
                if '漏洞类型' in line:
                    if '无' not in line and 'none' not in line.lower():
                        # 排除"如无则填"这种说明性文字
                        if '如无' not in line and 'if no' not in line.lower():
                            parts = line.split(':') if ':' in line else line.split('：')
                            if len(parts) > 1:
                                vuln_type = parts[1].strip()
                                if vuln_type and vuln_type != '无' and vuln_type.lower() != 'none':
                                    return True
        
        return False
    
    def update_chat_env(self, chat_env) -> ChatEnv:
        """更新全局环境"""
        # 检查是否真正执行了LLM探测
        llm_probing_executed = chat_env.env_dict.get('_llm_probing_executed', True)
        
        if self.vulnerability_found:
            chat_env.env_dict['vulnerability_detected'] = True
            chat_env.env_dict['vulnerability_info'] = self.vulnerability_details
            chat_env.env_dict['ideas'] = (
                f"⚠️ 发现漏洞！\n"
                f"合约: {self.vulnerability_details['contract']}\n"
                f"函数: {self.vulnerability_details['function']}\n"
                f"位置: 第 {self.vulnerability_details['index']}/{len(self.function_contexts)} 个函数\n\n"
                f"{self.vulnerability_details['response']}"
            )
        else:
            chat_env.env_dict['vulnerability_detected'] = False
            if llm_probing_executed:
                chat_env.env_dict['ideas'] = (
                    f"✅ 安全检查完成：探测了 {len(self.function_contexts)} 个函数，未发现可利用漏洞。"
                )
            else:
                chat_env.env_dict['ideas'] = (
                    f"⚠️ 未进行LLM探测：CKD阶段未生成可疑路径或数据缺失。"
                )
        
        # 保存二分类结果（用于CSV导出）
        chat_env.env_dict['binary_classification_result'] = {
            'has_vulnerability': self.vulnerability_found if llm_probing_executed else None,  # None表示未执行
            'solidity_file': chat_env.env_dict.get('solidity_file', ''),
            'llm_probing_executed': llm_probing_executed,
        }
        
        return chat_env



class ContractAnalysisET(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _discover_solidity_file(base_dir: str, hint: str = "") -> str:
        def _candidate(path: str) -> str | None:
            if path and os.path.isfile(path) and path.lower().endswith('.sol'):
                return path
            return None

        # Absolute or relative hint takes precedence
        if hint:
            expanded = os.path.abspath(os.path.join(base_dir, hint)) if not os.path.isabs(hint) else hint
            candidate = _candidate(expanded)
            if candidate:
                return candidate

        if base_dir and os.path.isdir(base_dir):
            for root, _, files in os.walk(base_dir):
                for filename in files:
                    if filename.lower().endswith('.sol'):
                        return os.path.join(root, filename)

        raise FileNotFoundError("No Solidity (.sol) file found for execution-trace analysis.")

    def update_phase_env(self, chat_env):
        base_dir = chat_env.env_dict.get('directory', '')
        hint = chat_env.env_dict.get('solidity_file', '')
        try:
            sol_file = self._discover_solidity_file(base_dir, hint)
        except FileNotFoundError as exc:
            # 回退方案：文件未找到
            fallback_src = chat_env.get_codes() or chat_env.env_dict.get('source_code', "")
            intel = {
                "mythril_trace": f"Solidity file missing: {exc}",
                "slither_logic": "",
                "source_code": fallback_src,
                "source_code_with_line_numbers": fallback_src,
                "mythril_vuln_count": 0,
                "mythril_severity_summary": "{}",
                "mythril_structured_report": "No analysis - file not found",
                "slither_cfg": "",
                "slither_function_summary": "",
                "slither_human_summary": "",
            }
            sol_file = hint or "<unknown>"
            log_visualize(f"**[ContractAnalysisET]** Unable to locate Solidity file. {exc}")
        else:
            try:
                # 运行增强的静态分析
                intel = run_contract_intel(sol_file, enhanced=True)
                
                # 详细日志输出
                vuln_count = intel.get('mythril_vuln_count', 0)
                severity_summary = intel.get('mythril_severity_summary', '{}')
                
                log_visualize(
                    f"**[ContractAnalysisET]** Successfully collected intelligence from {sol_file}\n"
                    f"  - Mythril: {vuln_count} vulnerabilities detected\n"
                    f"  - Severity Distribution: {severity_summary}\n"
                    f"  - Slither: Enhanced analysis completed"
                )
                
            except (ContractIntelError, FileNotFoundError, ValueError) as exc:
                # 回退方案：工具执行失败
                fallback_source = chat_env.get_codes() or chat_env.env_dict.get('source_code', "")
                intel = {
                    "mythril_trace": f"Failed to collect execution trace: {exc}",
                    "slither_logic": "",
                    "source_code": fallback_source,
                    "source_code_with_line_numbers": fallback_source,
                    "mythril_vuln_count": 0,
                    "mythril_severity_summary": "{}",
                    "mythril_structured_report": f"Analysis failed: {exc}",
                    "slither_cfg": "",
                    "slither_function_summary": "",
                    "slither_human_summary": "",
                }
                log_visualize(f"**[ContractAnalysisET][Error]** {exc}")

        # 更新phase环境，包含所有增强字段
        self.phase_env.update({
            "task": chat_env.env_dict['task_prompt'],
            "solidity_file": sol_file,
            # 原始字段（向后兼容）
            "execution_trace": intel.get('mythril_trace', ''),
            "function_logic": intel.get('slither_logic', ''),
            "source_code": intel.get('source_code', ''),
            # 新增：带行号的源代码
            "source_code_with_line_numbers": intel.get('source_code_with_line_numbers', intel.get('source_code', '')),
            # 新增的结构化字段
            "mythril_vuln_count": intel.get('mythril_vuln_count', 0),
            "mythril_severity_summary": intel.get('mythril_severity_summary', '{}'),
            "mythril_structured_report": intel.get('mythril_structured_report', ''),
            "slither_cfg": intel.get('slither_cfg', ''),
            "slither_function_summary": intel.get('slither_function_summary', ''),
            "slither_human_summary": intel.get('slither_human_summary', ''),
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            execution_trace = self.seminar_conclusion.split("<INFO>")[-1].strip()
        elif len(self.seminar_conclusion) > 0:
            execution_trace = self.seminar_conclusion.strip()
        else:
            execution_trace = ""

        chat_env.env_dict['execution_trace'] = execution_trace
        if not chat_env.env_dict['execution_trace']:
            chat_env.env_dict['execution_trace'] = "Execution trace unavailable."
        chat_env.env_dict['function_logic'] = self.phase_env.get('function_logic', '') or chat_env.env_dict.get('function_logic', '')
        chat_env.env_dict['source_code'] = self.phase_env.get('source_code', '') or chat_env.env_dict.get('source_code', '')
        
        # 设置 ideas 参数（模仿官方 ContractAnalysis 的实现）
        # 这对于后续阶段（如 ContractReviewComment）获取分析摘要非常重要
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        
        return chat_env

class ContractReviewComment(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas'],
             "execution_trace": chat_env.env_dict.get('execution_trace', ''),
             "function_logic": chat_env.env_dict.get('function_logic', '')})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['analysis_reports'] = self.seminar_conclusion
        return chat_env
    

class BugsSummary(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({
            "task": chat_env.env_dict['task_prompt'],
            "analysis_reports": chat_env.env_dict.get('analysis_reports', ''),
            "execution_trace": chat_env.env_dict.get('execution_trace', ''),
            "codes": chat_env.get_codes()
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        pass
        return chat_env


class isVul(Phase):
    """
    Binary vulnerability classification phase.
    Classifies smart contracts as either:
    - 1 (VULNERABLE): Contains exploitable security vulnerabilities
    - 0 (SECURE): No critical exploitable vulnerabilities
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        """
        Update phase environment with vulnerability analysis data.
        Uses the 'ideas' field which contains the vulnerability analysis from ContractReviewComment.
        """
        self.phase_env.update({
            "task": chat_env.env_dict['task_prompt'],
            "ideas": chat_env.env_dict.get('ideas', ''),
            "execution_trace": chat_env.env_dict.get('execution_trace', ''),
            "analysis_reports": chat_env.env_dict.get('analysis_reports', ''),
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        """
        Update chat environment with binary classification result.
        Extracts the classification (1 or 0) from seminar conclusion.
        """
        # Extract the binary classification from the conclusion
        classification = self.seminar_conclusion.strip()
        
        # Try to extract just the number (1 or 0) from the conclusion
        if '1' in classification:
            chat_env.env_dict['vulnerability_classification'] = 1
            chat_env.env_dict['is_vulnerable'] = True
        elif '0' in classification:
            chat_env.env_dict['vulnerability_classification'] = 0
            chat_env.env_dict['is_vulnerable'] = False
        else:
            # Default to vulnerable if unclear
            chat_env.env_dict['vulnerability_classification'] = 1
            chat_env.env_dict['is_vulnerable'] = True
        
        # Store the full classification report
        chat_env.env_dict['vulnerability_classification_report'] = self.seminar_conclusion
        
        return chat_env

    
class TestBugsSummary(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        chat_env.generate_images_from_codes()
        (exist_bugs_flag, test_reports) = chat_env.exist_bugs()
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "ideas": chat_env.env_dict['ideas'],
                               "test_reports": test_reports,
                               "exist_bugs_flag": exist_bugs_flag})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['error_summary'] = self.seminar_conclusion
        chat_env.env_dict['test_reports'] = self.phase_env['test_reports']

        return chat_env

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        if "ModuleNotFoundError" in self.phase_env['test_reports']:
            chat_env.fix_module_not_found_error(self.phase_env['test_reports'])
            pip_install_content = ""
            for match in re.finditer(r"No module named '(\S+)'", self.phase_env['test_reports'], re.DOTALL):
                module = match.group(1)
                pip_install_content += "{}\n```{}\n{}\n```\n".format(
                    "cmd", "bash", f"pip install {module}")
            self.seminar_conclusion = "nothing need to do"
        else:
            self.seminar_conclusion = \
                self.chatting(chat_env=chat_env,
                              task_prompt=chat_env.env_dict['task_prompt'],
                              need_reflect=need_reflect,
                              assistant_role_name=self.assistant_role_name,
                              user_role_name=self.user_role_name,
                              phase_prompt=self.phase_prompt,
                              phase_name=self.phase_name,
                              assistant_role_prompt=self.assistant_role_prompt,
                              user_role_prompt=self.user_role_prompt,
                              chat_turn_limit=chat_turn_limit,
                              placeholders=self.phase_env)
        chat_env = self.update_chat_env(chat_env)
        return chat_env
    
class ContractModification(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update({
            "task": chat_env.env_dict['task_prompt'],
            "ideas": chat_env.env_dict['ideas'],
            "test_reports": chat_env.env_dict['test_reports'],
            "error_summary": chat_env.env_dict['error_summary']
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        if "```".lower() in self.seminar_conclusion.lower():
            chat_env.update_codes(self.seminar_conclusion)
            chat_env.rewrite_codes(
                "Test #" + " Finished")
        return chat_env


class ArithmeticDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class ReentrancyDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class UncheckedSendDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


# class DelegatecallDetector(Phase):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def update_phase_env(self, chat_env):
#         self.phase_env.update(
#             {"task": chat_env.env_dict['task_prompt'],
#              "ideas": chat_env.env_dict['ideas']})

#     def update_chat_env(self, chat_env) -> ChatEnv:
#         if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
#             chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
#                 "<INFO>")[-1].lower().replace(".", "").strip()
#         elif len(self.seminar_conclusion) > 0:
#             chat_env.env_dict['ideas'] = self.seminar_conclusion
#         else:
#             chat_env.env_dict['ideas'] = "I have no idea"
#         return chat_env


class TODDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class TimeStampManipulationDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class TXRelianceDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class SuicideDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class GasLimitDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class PredictableRandDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    
    
class PriceManipulationDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    

class DataCorruptionDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class WithdrawalFunctionDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class LackAuthorizationDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class DataInconsistencyDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class HashCollisionDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class UninitializedReturnVariableDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class MisdeclaredConstructorDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    

class MissingOnlyOwnerDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class MisuseMsgValueDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class PrecisionLossDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class RedundantConditionalDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class OracleDependencyDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class OwnershipHijackingDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class CentralizationRiskDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class FundingCalculationDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class FlashLoanDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class MappingGetterDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    

class GetterFunctionDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    

class UnnecessaryComparisonDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    
    
class InconsistentInitializationDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class SourceSwappingDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class SignatureVerificationDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    

class OrderInitializationDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class ImpracticalityMatchDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class InconsistentTokensDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    

class PartialWithdrawalsDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class FallbackFunctionDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    

class UnlimitedTokenDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env


class InputValidationDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env

class UncheckedLowLevelCallDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env
    
class DoSDetector(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env.update(
            {"task": chat_env.env_dict['task_prompt'],
             "ideas": chat_env.env_dict['ideas']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0 and "<INFO>" in self.seminar_conclusion:
            chat_env.env_dict['ideas'] = self.seminar_conclusion.split(
                "<INFO>")[-1].lower().replace(".", "").strip()
        elif len(self.seminar_conclusion) > 0:
            chat_env.env_dict['ideas'] = self.seminar_conclusion
        else:
            chat_env.env_dict['ideas'] = "I have no idea"
        return chat_env