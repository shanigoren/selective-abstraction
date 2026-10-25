# Derived from https://github.com/stanford-oval/spinach (Apache-2.0).
# This release preserves local Selective Abstraction integration changes,
# including evaluator generation-parameter plumbing.
import json
import sys

from typing import List, Dict
from chainlite import chain, get_logger, llm_generation_chain, pprint_chain
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnablePassthrough
import os
import re



from spinach_agent.parser_state import (
    Action,
    PartToWholeParserState,
    SparqlQuery,
    state_to_dict,
    state_to_string,
)
from spinach_agent.parser_utils import (
    BaseParser,
    execute_sparql_object,
    get_prune_edges_chain,
    modified_llm_generation_chain,
    parse_string_to_json,
    sparql_string_to_sparql_object,
)
from wikidata_utils import (
    SparqlExecutionStatus,
    get_outgoing_edges,
    search_span,
)
from wikidata_utils import get_property_examples

logger = get_logger(__name__)

wikidata_data_type_to_description_map = {
    # "string": "",
    "external-id": "Represents an identifier used in an external system.",
    "url": "Represents a link to an external resource or site. Can be converted to string using STR()",
    "commonsMedia": "References to files (like images, audio) on Wikimedia Commons.",
    "geo-shape": "Reference to map data files on Wikimedia Commons.",
    "tabular-data": "Reference to tabular data files on Wikimedia Commons.",
    "math": "Stores and displays formatted mathematical formulas.",
    "musical-notation": "Stores and displays musical scores as generated images in .png format.",
}


def format_wikidata_search_result(arr):
    ret = []
    for item in arr:
        label = item.get(
            "label", ""
        )  # Get the label or an empty string if not available
        id = item.get("id", "")  # Get the id or an empty string if not available
        description = item.get(
            "description", ""
        )  # Get the description or an empty string if not available

        display_string = f"{label} ({id}): {description}"
        if "datatype" in item:
            data_type = item["datatype"]
            if data_type in wikidata_data_type_to_description_map:
                data_type += f" ({wikidata_data_type_to_description_map[data_type]})"
            display_string += f"\tData Type: {data_type}"

        ret.append(display_string)
    return "\n".join(ret)


@chain
async def json_to_string(j: dict) -> str:
    return json.dumps(j, indent=2, ensure_ascii=False)


@chain
async def json_to_action(action_dict: dict) -> Action:
    # if action_dict is a list:
    if isinstance(action_dict, list):
        # print(f"Warning: action_dict is a list")
        action_dict_orig = action_dict
        action_dict = action_dict[-1]
    if isinstance(action_dict, str):
        print(f"Warning: action_dict is a string: {action_dict}")
    thought = action_dict["thought"]
    action_name = action_dict["action_name"]
    action_argument = action_dict["action_argument"]

    if action_name == "execute_sparql":
        assert action_argument, action_dict

    return Action(
        thought=thought,
        action_name=action_name,
        action_argument=action_argument,
    )


def display_actions(actions):
    # truncate actions to the most recent 10
    actions = actions[-10:]
    
    # Create a dictionary to track the most recent action of each type+argument
    unique_actions = {}
    
    # Process actions to keep only the most recent instance of each action type+argument
    for a in actions:
        # Skip any empty lists or non-Action objects
        if not a or not hasattr(a, 'action_name'):
            continue
        
        # Create a key based on action_name and action_argument
        action_key = f"{a.action_name}:{a.action_argument}"
        
        # Store this action (will replace any previous action with same key)
        unique_actions[action_key] = a
    
    # Get the list of unique actions and sort by original order
    # This is a simple approach - for more sophisticated ordering,
    # we would need to track original indices
    unique_action_list = list(unique_actions.values())
    
    # Now format the action history
    action_history = []
    for i, a in enumerate(unique_action_list):
        include_observation = True
        if i < len(unique_action_list) - 2 and a.action_name in [
            "get_wikidata_entry",
            "search_wikidata",
        ]:
            include_observation = False
        if a.action_name == "stop":
            include_observation = False
        action_history.append(a.to_jinja_string(include_observation))
    
    return action_history


class PartToWholeParser(BaseParser):
    @classmethod
    def initialize(
        cls,
        engine: str,
        max_response_tokens: int = 700,
        reasoning_effort: str = None,
        temperature: float = None,
        top_p: float = None,
        # top_k: int = None,
    ):
        if reasoning_effort in ("none", "omit"):
            reasoning_effort = None

        @chain
        async def initialize_state(input):
            is_entity_info = cls.is_entity_info_format(input["question"])
            return PartToWholeParserState(
                question=input["question"],
                conversation_history=input["conversation_history"],
                engine=engine,
                action_counter=0,
                actions=[],
                response="",
                is_entity_info_mode=is_entity_info,
                generated_sparqls=[],
                final_sparql=None
            )

        # build the graph
        graph = StateGraph(PartToWholeParserState)
        
        # Add all nodes
        graph.add_node("start", lambda x: {})
        graph.add_node("controller", cls.controller)
        # graph.add_node("entity_counter", cls.entity_counter)
        graph.add_node("execute_sparql", cls.execute_sparql)
        graph.add_node("get_wikidata_entry", cls.get_wikidata_entry)
        graph.add_node("search_wikidata", cls.search_wikidata)
        graph.add_node("get_property_examples", cls.get_property_examples)
        # graph.add_node("report_category", cls.report_category)
        # graph.add_node("report_category_count", cls.report_category_count)
        # graph.add_node("report_matching_count", cls.report_matching_count)
        graph.add_node("reporter", cls.reporter)
        graph.add_node("stop", cls.stop)

        # Set entry point
        graph.set_entry_point("start")

        # Create conditional branch from start node to either controller or entity_counter
        def start_router(state):
            # if state["is_entity_info_mode"]:
            #     return "entity_counter"
            # else:
            #     return "controller"
            return "controller"
        
        graph.add_conditional_edges("start", start_router)
        
        # Routing from controller node
        def controller_router(state):
            if "active_action" in state and state["active_action"]:
                current_action = state["active_action"]
                
                if current_action.action_name == "stop":
                    return "stop"
                # elif current_action.action_name in ["execute_sparql", "get_wikidata_entry", "search_wikidata", "get_property_examples", "report_category", "report_category_count", "report_matching_count"]:
                elif current_action.action_name in ["execute_sparql", "get_wikidata_entry", "search_wikidata", "get_property_examples"]:
                    return current_action.action_name
            
            # Default case - stay in controller
            return "controller"
        
        graph.add_conditional_edges("controller", controller_router)
        
        # # Routing from entity_counter node
        # def entity_counter_router(state):
        #     if "active_action" in state and state["active_action"]:
        #         current_action = state["active_action"]
                
        #         if current_action.action_name == "stop":
        #             return "stop"
        #         elif current_action.action_name in ["execute_sparql", "get_wikidata_entry", "search_wikidata", "get_property_examples", "report_category", "report_category_count", "report_matching_count"]:
        #             return current_action.action_name
            
        #     # Default case - stay in entity_counter
        #     return "entity_counter"
        
        # graph.add_conditional_edges("entity_counter", entity_counter_router)
        
        # Routes from actions back to controllers
        def action_to_controller_router(state):
            # # Clear any active_action when returning to controller
            # if state["is_entity_info_mode"]:
            #     return "entity_counter"
            # else:
            #     return "controller"
            return "controller"
        
        for action_node in ["execute_sparql", "get_wikidata_entry", "search_wikidata", "get_property_examples"]:
            # for action_node in ["execute_sparql", "get_wikidata_entry", "search_wikidata", "get_property_examples", "report_category", "report_category_count", "report_matching_count"]:
            graph.add_conditional_edges(action_node, action_to_controller_router)
        
        # Final steps
        graph.add_edge("stop", "reporter")
        graph.add_edge("reporter", END)

        # Initialize chains
        cls.controller_chain = (
            {
                "input": modified_llm_generation_chain(
                    template_file="controller.prompt",
                    engine=engine,
                    max_tokens=max_response_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    # top_k=top_k,
                    keep_indentation=True,
                    reasoning_effort=reasoning_effort,
                )
            }
            | modified_llm_generation_chain(
                template_file="format_actions.prompt",
                engine=engine,
                max_tokens=max_response_tokens,
                keep_indentation=True,
                output_json=True,
                # reasoning_effort="low",
            )
            | parse_string_to_json
            | json_to_action
        )
        
        cls.sparql_chain = sparql_string_to_sparql_object | execute_sparql_object
        cls.prune_edges_chain = get_prune_edges_chain(engine=engine) | json_to_string
        cls.reporter_chain = modified_llm_generation_chain(
            template_file="conversation_reporter.prompt",
            engine=engine,
            # max_tokens=2000,
            max_tokens=max_response_tokens,
            temperature=0.0,
            keep_indentation=True,
            reasoning_effort="low",
        )
        cls.language_detection_chain = modified_llm_generation_chain(
            template_file="language_detection.prompt",
            engine=engine,
            # max_tokens=50,
            max_tokens=max_response_tokens,
            temperature=0.0,
            top_p=0.9,
            keep_indentation=True,
            reasoning_effort="low",
        )

        compiled_graph = graph.compile()
        cls.runnable = initialize_state | compiled_graph
        logger.info("Finished initializing the graph.")
        print("Finished initializing the graph.")

    @staticmethod
    def is_entity_info_format(question):
        """Check if the question follows the entity info format: [entity] [information about the entity]."""
        pattern = r'^\[.+\] \[.+\]\.$'
        return bool(re.match(pattern, question.strip()))

    @staticmethod
    def get_current_action(state):
        return state["actions"][-1]

    @staticmethod
    def route_action(state):
        """Route to the next node based on the current action."""
        if not state["actions"]:
            # # No actions yet, stay in the same controller
            # if state["is_entity_info_mode"]:
            #     return "entity_counter"
            # else:
            #     return "controller"
            return "controller"
            
        current_action = state["actions"][-1]
        
        if current_action.action_name == "stop":
            return "stop"
        
        # Route to the appropriate action handler
        if current_action.action_name in ["execute_sparql", "get_wikidata_entry", "search_wikidata", "get_property_examples"]:
            # if current_action.action_name in ["execute_sparql", "get_wikidata_entry", "search_wikidata", "get_property_examples", "report_category"]:
            return current_action.action_name
        
        # # Default - stay in the same controller
        # if state["is_entity_info_mode"]:
        #     return "entity_counter"
        # else:
        #     return "controller"
        return "controller"

    @staticmethod
    @chain
    async def controller(state):
        """Standard controller for processing regular questions."""
        action_history = display_actions(state["actions"])
        action = await PartToWholeParser.controller_chain.ainvoke( # this is where the controller decides which action to take. observation will be empty and added when the action is executed.
            {
                "conversation_history": state["conversation_history"],
                "question": state["question"],
                "action_history": action_history
            }
        )
        # Set the active_action field instead of modifying the actions list directly
        return {"active_action": action, "action_counter": 1}

    # @staticmethod
    # @chain
    # async def entity_counter(state):
    #     """Handler for entity-information format processing."""
    #     action_history = display_actions(state["actions"])
    #     if len(action_history) > 0:
    #         logger.info(f"{action_history[-1].strip()}\n"+  '-'*100)
    #     action = await PartToWholeParser.entity_counter_chain.ainvoke(
    #         {
    #             "conversation_history": state["conversation_history"],
    #             "question": state["question"],
    #             "action_history": action_history
    #         }
    #     )
    #     # Set the active_action field instead of modifying the actions list directly
    #     return {"active_action": action, "action_counter": 1}
        
    @staticmethod
    @chain
    async def execute_sparql(state):
        curr_action = state["active_action"]
        assert curr_action.action_name == "execute_sparql"
        sparql = curr_action.action_argument
        # create the sparql object, and add it to the list
        sparql_obj = await sparql_string_to_sparql_object.ainvoke(sparql)
        if sparql_obj.is_valid:
            state["generated_sparqls"].append(sparql_obj)
            sparql_obj = await execute_sparql_object.ainvoke(sparql_obj)
            
            # Check if query timed out
            if sparql_obj.has_timed_out():
                # Add a note about the timeout
                curr_action.observation = "TIMEOUT: The query execution timed out. This typically happens with complex queries or queries that would return a large number of results."
                # curr_action.observation += "\nI need to consider a complement approach (count the items that don't fulfill the condition) or breaking down the query into smaller parts and then combine the results."
            else:
                curr_action.observation = sparql_obj.results_in_table_format()
                curr_action.observation_markdown = sparql_obj.results_in_markdown_format()
        else:
            curr_action.observation = (
                f"Error: SPARQL is not valid. Please fix and resubmit: {sparql_obj.sparql}"
            )

        # Add the action to the actions list and clear active_action
        return {"actions": curr_action, "active_action": None}

    @staticmethod
    @chain
    async def get_wikidata_entry(state):
        curr_action = state["active_action"]
        assert curr_action.action_name == "get_wikidata_entry"
        outgoing_edges = get_outgoing_edges(curr_action.action_argument, compact=False)
        if outgoing_edges == None:
            # No outgoing edges found. Possibly invalid entity ID or other error.
            curr_action.observation = f"No outgoing edges found for entity: {curr_action.action_argument}. Please verify that the entity ID is correct."
        else:
            curr_action.observation = outgoing_edges

        # Add the action to the actions list and clear active_action
        return {"actions": curr_action, "active_action": None}

    @staticmethod
    @chain
    async def search_wikidata(state):
        curr_action = state["active_action"]
        assert curr_action.action_name == "search_wikidata"
        query = curr_action.action_argument
        if "print_error_detail" in query:
            curr_action.observation = f"Detailed error: {query}"
            return {"actions": curr_action, "active_action": None}

        curr_action.observation = format_wikidata_search_result(search_span(query, return_full_results=True))
        return {"actions": curr_action, "active_action": None}

    @staticmethod
    @chain
    async def get_property_examples(state):
        curr_action = state["active_action"]
        assert curr_action.action_name == "get_property_examples"
        pid = curr_action.action_argument
        examples = get_property_examples(pid)
        curr_action.observation = examples
        return {"actions": curr_action, "active_action": None}

    @staticmethod
    @chain
    async def stop(state):
        curr_action = state["active_action"]
        assert curr_action.action_name == "stop"
        if curr_action.action_argument == '':
            curr_action.action_argument = curr_action.thought
        curr_action.observation = f"Agent has stopped working on query: {state['question']}"
        
        # Check if we need to find the final sparql
        if len(state["generated_sparqls"]) > 0:
            # Find the final sparql query
            # Let's get the last one that has results by default
            sparql_index = len(state["generated_sparqls"]) - 1
            while sparql_index >= 0:
                # if state["generated_sparqls"][sparql_index] is a list, print the length of the list
                if state["generated_sparqls"][sparql_index] != []:
                    if state["generated_sparqls"][sparql_index].has_results():
                        state["final_sparql"] = state["generated_sparqls"][sparql_index]
                        break
                sparql_index -= 1
            return {"actions": curr_action, "active_action": None, "final_sparql": state["final_sparql"]}
        
        return {"actions": curr_action, "active_action": None}

    @staticmethod
    @chain
    async def reporter(state):
        from spinach_agent.parser_utils import search_final_sparql
        import logging

        # Make sure we have no lingering active_action
        state["active_action"] = None
        
        # Create a deduplicated action history for display purposes
        action_history = display_actions(state["actions"])
        
        # Check if we're in entity info mode
        # if state["is_entity_info_mode"]:
            # # Process entity info results
            # # Extract entity and information
            # match = re.match(r'^\[(.*?)\] \[(.*?)\]\.$', state["question"].strip())
            # entity = match.group(1) if match else "unknown entity"
            # information = match.group(2) if match else "unknown information"
            
            # # Initialize values
            # category_type = None
            # category_count = None
            # matching_count = None
            
            # # Find the values from the explicit report actions - process only the latest instances
            # seen_action_types = set()
            # report_actions = {
            #     "report_category": None,
            #     "report_category_count": None,
            #     "report_matching_count": None
            # }
            
            # # Process actions in reverse order to get the most recent values
            # for action in reversed(state["actions"]):
            #     if not action: continue
                
            #     action_type = getattr(action, 'action_name', None)
            #     if not action_type or action_type not in report_actions or action_type in seen_action_types:
            #         continue
                    
            #     # Store the most recent instance of each report action
            #     report_actions[action_type] = action
            #     seen_action_types.add(action_type)
                
            #     # Stop once we've found all three types
            #     if len(seen_action_types) >= 3:
            #         break
            
            # # Extract values from the most recent report actions
            # if report_actions["report_category"]:
            #     category_type = getattr(report_actions["report_category"], 'entity_category', None)
            #     if not category_type:
            #         logging.warning("Missing entity_category value in report_category action")
            # else:
            #     logging.warning("Missing report_category action")
                
            # if report_actions["report_category_count"]:
            #     category_count = getattr(report_actions["report_category_count"], 'category_count', None)
            #     if not category_count:
            #         logging.warning("Missing category_count value in report_category_count action")
            # else:
            #     logging.warning("Missing report_category_count action")
                
            # if report_actions["report_matching_count"]:
            #     matching_count = getattr(report_actions["report_matching_count"], 'matching_count', None)
            #     if not matching_count:
            #         logging.warning("Missing matching_count value in report_matching_count action")
            # else:
            #     logging.warning("Missing report_matching_count action")
            
            # # Create a summary for the response
            # summary = f"Entity: {entity}\n"
            # summary += f"Information: {information}\n"
            # summary += f"Category: {category_type or 'Unknown'}\n"
            # summary += f"Category Count: {category_count or 'Unknown'}\n"
            # summary += f"Matching Count: {matching_count or 'Unknown'}"
            
            # return {"response": summary}
        # else:   
        # Standard mode - extract final SPARQL query
        final_sparql = state.get("final_sparql", None)
        if final_sparql is None and state["generated_sparqls"]:
            final_sparql = search_final_sparql(state)

        language = await PartToWholeParser.language_detection_chain.ainvoke(
            {
                "question": state["question"],
                "sparql": final_sparql.sparql if final_sparql else None,
            }
        )
        
        response = await PartToWholeParser.reporter_chain.ainvoke(
            {
                "language": language,
                "sparql": final_sparql.sparql if final_sparql else None,
                "question": state["question"],
                "action_history": action_history,
                "conversation_history": state["conversation_history"],
            }
        )
        
        return {"response": response}

    # @staticmethod
    # @chain
    # async def report_category(state):
    #     """Handler for the report_category action."""
    #     curr_action = state["active_action"]
    #     assert curr_action.action_name == "report_category"
    #     # Store the category in the action's entity_category field
    #     curr_action.entity_category = curr_action.action_argument
    #     curr_action.observation = f"Category identified: {curr_action.action_argument}"
    #     return {"actions": curr_action, "active_action": None}
        
    # @staticmethod
    # @chain
    # async def report_category_count(state):
    #     """Handler for the report_category_count action."""
    #     curr_action = state["active_action"]
    #     assert curr_action.action_name == "report_category_count"
    #     # Store the count in a custom field
    #     curr_action.category_count = curr_action.action_argument
    #     curr_action.observation = f"Category count recorded: {curr_action.action_argument}"
    #     return {"actions": curr_action, "active_action": None}
        
    # @staticmethod
    # @chain
    # async def report_matching_count(state):
    #     """Handler for the report_matching_count action."""
    #     curr_action = state["active_action"]
    #     assert curr_action.action_name == "report_matching_count"
    #     # Store the count in a custom field
    #     curr_action.matching_count = curr_action.action_argument
    #     curr_action.observation = f"Matching count recorded: {curr_action.action_argument}"
    #     return {"actions": curr_action, "active_action": None}
