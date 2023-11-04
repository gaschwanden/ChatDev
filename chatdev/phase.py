import os
import re
from abc import ABC, abstractmethod

from camel.agents import RolePlaying
from camel.messages import ChatMessage
from camel.typing import TaskType, ModelType
from chatdev.chat_env import ChatEnv
from chatdev.statistics import get_info
from chatdev.utils import log_and_print_online, log_arguments


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
        self.timeout_seconds = 1.0
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
            model_type=ModelType.GPT_3_5_TURBO,
            placeholders=None,
            chat_turn_limit=2
    ) -> str:
        """

        Args:
            chat_env: global chatchain environment TODO: only for employee detection, can be deleted
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
            model_type=model_type,
        )

        # log_and_print_online("System", role_play_session.assistant_sys_msg)
        # log_and_print_online("System", role_play_session.user_sys_msg)

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
                log_and_print_online(role_play_session.assistant_agent.role_name,
                                     conversation_meta + "[" + role_play_session.user_agent.system_message.content + "]\n\n" + assistant_response.msg.content)
                if role_play_session.assistant_agent.info:
                    seminar_conclusion = assistant_response.msg.content
                    break
                if assistant_response.terminated:
                    break

            if isinstance(user_response.msg, ChatMessage):
                # here is the result of the second interaction, which may be used to start the next chat turn
                log_and_print_online(role_play_session.user_agent.role_name,
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

        log_and_print_online("**[Seminar Conclusion]**:\n\n {}".format(seminar_conclusion))
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
        # elif phase_name in [PhaseType.BRAINSTORMING]:
        #     question = """Conclude three most creative and imaginative brainstorm ideas from the whole discussion, in the format: "1) *; 2) *; 3) *; where '*' represents a suggestion." """
        elif phase_name == "LanguageChoose":
            question = """Conclude the programming language being discussed for software development, in the format: "*" where '*' represents a programming language." """
        elif phase_name == "EnvironmentDoc":
            question = """According to the codes and file format listed above, write a requirements.txt file to specify the dependencies or packages required for the project to run properly." """
        elif phase_name == "EngagementAnalysis":
            question = """Reflect on the recent customer engagement, identifying strengths and areas for improvement." """
        elif phase_name == "ActionItemIdentification":
            question = """Identify specific action items to improve future engagements based on the analysis and limit them to the top 10 itmes." """

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
                          model_type=self.model_type)
        chat_env = self.update_chat_env(chat_env)
        return chat_env

class Assignment(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        # Update the phase environment with details relevant to assignment of tasks.
        # You may want to retrieve a list of agents and tasks from `chat_env`.
        self.phase_env.update({
            "agents": chat_env.env_dict.get('agents', []),
            "tasks": chat_env.env_dict.get('tasks', [])
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        # This method would be responsible for assigning the tasks to the agents
        # within the chat environment. The specifics would depend on your application's logic.

        # Check if tasks and agents are available to assign
        if not self.phase_env['tasks'] or not self.phase_env['agents']:
            raise ValueError("Tasks or agents information is missing.")

        # Proceed with assigning tasks to agents
        for agent in self.phase_env['agents']:
            # This is a simplified example of how you might assign tasks.
            # The actual logic could be more complex, depending on your needs.
            assigned_task = self.get_task_for_agent(agent)
            chat_env.assign_task(agent, assigned_task)

        # Assuming a method exists to log the assignments.
        log_and_print_online("**[Task Assignments]**:\n\n {}".format(
            self.generate_assignment_summary()))
        
        return chat_env

    def get_task_for_agent(self, agent):
        # This method would contain the logic to determine which task to assign to which agent.
        # For now, we'll just return a placeholder task.
        return "Complete customer follow-up"

    def generate_assignment_summary(self):
        # This would create a summary of the assignments that have been made.
        # For now, it's just a placeholder text.
        return "All agents have been assigned tasks."


class PerformanceMetricsReview(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        # Here, you'd extract any relevant performance metrics from the chat environment
        # that may need to be reviewed in this phase.
        self.phase_env.update({
            "performance_metrics": chat_env.env_dict.get('performance_metrics', []),
            "improvements": chat_env.env_dict.get('implemented_improvements', [])
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        # In this method, performance metrics would be analyzed to assess the
        # effectiveness of the improvements that have been implemented.
        if not self.phase_env['performance_metrics']:
            raise ValueError("No performance metrics available for review.")

        # Assuming a method exists to analyze the performance metrics.
        performance_analysis = self.analyze_performance(self.phase_env['performance_metrics'])

        # Update the chat environment with the analysis results.
        chat_env.update_performance_analysis(performance_analysis)

        # Log and print the performance review summary.
        log_and_print_online("**[Performance Metrics Review]**:\n\n {}".format(
            self.generate_performance_review_summary(performance_analysis)))

        return chat_env

    def analyze_performance(self, performance_metrics):
        # This method contains the logic for analyzing performance metrics.
        # The specifics of this method would depend on what metrics are collected
        # and how you determine effectiveness.
        # This is a placeholder for illustrative purposes.
        return {
            "metric_summary": "Summary of key metrics.",
            "improvement_outcomes": "Analysis of the outcomes related to the improvements."
        }

    def generate_performance_review_summary(self, performance_analysis):
        # This function would create a summary of the performance review based on
        # the analysis performed in the `analyze_performance` method.
        return (
            f"{performance_analysis['metric_summary']}\n"
            f"Effectiveness of Improvements:\n"
            f"{performance_analysis['improvement_outcomes']}"
        )


class CustomerFeedbackIntegration(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        # Extract feedback and improvements from the chat environment.
        # These would be used to ensure alignment with customer expectations.
        self.phase_env.update({
            "customer_feedback": chat_env.env_dict.get('customer_feedback', []),
            "planned_improvements": chat_env.env_dict.get('planned_improvements', [])
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        # This method integrates customer feedback with the planned improvements.
        if not self.phase_env['customer_feedback']:
            raise ValueError("No customer feedback available for integration.")

        # Assuming a method exists to integrate feedback into planning.
        chat_env.integrate_feedback(self.phase_env['customer_feedback'],
                                    self.phase_env['planned_improvements'])

        # Assuming there's a method to log and print the feedback integration process.
        log_and_print_online("**[Customer Feedback Integration]**:\n\n {}".format(
            self.generate_feedback_integration_summary()))

        return chat_env

    def generate_feedback_integration_summary(self):
        # Summarize the process of integrating customer feedback.
        # This would include details on how the feedback is affecting planning.
        return "Customer feedback has been reviewed and integrated into the improvement plans."


class DemandAnalysis(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        pass

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0:
            chat_env.env_dict['modality'] = self.seminar_conclusion.split("<INFO>")[-1].lower().replace(".", "").strip()
        return chat_env
    
class EngagementAnalysis(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        # For EngagementAnalysis, the details to be included in phase_env would be specific to customer engagement.
        # There might not be a GUI component here, as it seems to focus on analysis of interactions rather than software design.
        self.phase_env.update({
            "strengths": chat_env.env_dict.get('strengths', ''),
            "improvements": chat_env.env_dict.get('improvements', '')
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        # This method would implement any changes to the chat environment based on the results of the engagement analysis.
        # Assuming the ChatEnv class has methods to handle the updating process based on the provided configuration.
        chat_env.update_engagement_analysis(self.phase_env)  # This is hypothetical and depends on your ChatEnv class definition.
        # Log the updated environment for debugging or record-keeping purposes.
        log_and_print_online("**[Engagement Feedback]**:\n\n {}".format(self._generate_feedback_summary(chat_env)))
        return chat_env

    def _generate_feedback_summary(self, chat_env):
        # Generate a summary of the engagement feedback, based on the chat environment.
        # This might involve formatting the strengths and areas for improvement identified during the phase.
        strengths = chat_env.env_dict.get('strengths', 'No strengths recorded.')
        improvements = chat_env.env_dict.get('improvements', 'No improvements recorded.')
        return f"Strengths identified:\n{strengths}\n\nAreas for improvement:\n{improvements}"

    
class ActionItemIdentification(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        pass

    def update_chat_env(self, chat_env) -> ChatEnv:
        if len(self.seminar_conclusion) > 0:
            chat_env.env_dict['modality'] = self.seminar_conclusion.split("<INFO>")[-1].lower().replace(".", "").strip()
        return chat_env

class EngagementAnalysis(Phase):
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
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "gui": gui})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.update_codes(self.seminar_conclusion)
        if len(chat_env.codes.codebooks.keys()) == 0:
            raise ValueError("No Valid Codes.")
        chat_env.rewrite_codes()
        log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        return chat_env


class ArtDesign(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        self.phase_env = {"task": chat_env.env_dict['task_prompt'],
                          "language": chat_env.env_dict['language'],
                          "codes": chat_env.get_codes()}

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.proposed_images = chat_env.get_proposed_images_from_message(self.seminar_conclusion)
        log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
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
        chat_env.rewrite_codes()
        # chat_env.generate_images_from_codes()
        log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
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
        chat_env.rewrite_codes()
        log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        return chat_env

class ChallengesOpportunities(Phase):
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
        chat_env.rewrite_codes()
        log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        return chat_env

class ImprovementPlanning(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        # In this method, you would update the phase environment with any
        # information relevant to the improvement planning. If `chat_env` contains
        # specific improvements to implement, they would be extracted here.
        self.phase_env.update({
            "improvements": chat_env.env_dict.get('identified_improvements', [])
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        # Here, the chat environment would be updated to reflect the plan for
        # implementing the identified improvements.
        # You might define a method to create an action plan based on the improvements.
        if not self.phase_env['improvements']:
            raise ValueError("No identified improvements to plan for.")

        # Let's assume there's a method to plan improvements in your environment.
        for improvement in self.phase_env['improvements']:
            chat_env.plan_improvement(improvement)

        # Assuming a method to log the planning details.
        log_and_print_online("**[Improvement Planning]**:\n\n {}".format(
            self.generate_planning_summary()))

        return chat_env

    def generate_planning_summary(self):
        # Create a summary of the improvement plan that's been constructed.
        # This would likely be a more complex function in a real scenario.
        return "Improvement implementation plans have been created for all identified areas."


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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env
    
class AnalysisSalesProcess(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env

class ChallengesOpportunities(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env

class AnalysisSalesProcess(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env
    
class Implementation(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env
    
class StrategyDevelopment(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env
    
class LongTermOptimization(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env
    
class MonitoringAdjustment(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env

class ReportingDocumentation(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env
class LongTermOptimization(Phase):
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
        self.phase_env['modification_conclusion'] = self.seminar_conclusion
        return chat_env

class CodeReviewHuman(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_phase_env(self, chat_env):
        print(
            f"You can participate in the development of the software {chat_env.env_dict['task_prompt']}. Please input your feedback. (\"End\" to quit the involvement.)")
        provided_comments = input()
        self.phase_env.update({"task": chat_env.env_dict['task_prompt'],
                               "modality": chat_env.env_dict['modality'],
                               "ideas": chat_env.env_dict['ideas'],
                               "language": chat_env.env_dict['language'],
                               "codes": chat_env.get_codes(),
                               "comments": provided_comments})

    def update_chat_env(self, chat_env) -> ChatEnv:
        if "```".lower() in self.seminar_conclusion.lower():
            chat_env.update_codes(self.seminar_conclusion)
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
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
        log_and_print_online("**[Test Reports]**:\n\n{}".format(test_reports))

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['error_summary'] = self.seminar_conclusion
        chat_env.env_dict['test_reports'] = self.phase_env['test_reports']

        return chat_env

    def execute(self, chat_env, chat_turn_limit, need_reflect) -> ChatEnv:
        self.update_phase_env(chat_env)
        if "ModuleNotFoundError" in self.phase_env['test_reports']:
            chat_env.fix_module_not_found_error(self.phase_env['test_reports'])
            log_and_print_online(
                f"Software Test Engineer found ModuleNotFoundError:\n{self.phase_env['test_reports']}\n")
            pip_install_content = ""
            for match in re.finditer(r"No module named '(\S+)'", self.phase_env['test_reports'], re.DOTALL):
                module = match.group(1)
                pip_install_content += "{}\n```{}\n{}\n```\n".format("cmd", "bash", f"pip install {module}")
                log_and_print_online(f"Programmer resolve ModuleNotFoundError by:\n{pip_install_content}\n")
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
            chat_env.rewrite_codes()
            log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
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
        log_and_print_online("**[Software Info]**:\n\n {}".format(get_info(chat_env.env_dict['directory'],self.log_filepath)))
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
