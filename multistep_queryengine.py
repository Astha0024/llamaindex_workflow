# Break complex questions into sequential sub questions
import os
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.workflow import Event
from typing import Dict, List, Any
from llama_index.core.schema import NodeWithScore

from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)

from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
)

from llama_index.core.schema import QueryBundle, TextNode

from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.core import Settings
from llama_index.core.llms import LLM

from typing import cast

from llama_index.core import SimpleDirectoryReader

from llama_index.llms.openai import OpenAI

from llama_index.core import VectorStoreIndex

# Define event
class QueryMultiStepEvent(Event):
    """
    Event containing results of a multi-step query process.

    Attributes:
        nodes (List[NodeWithScore]): List of nodes with their associated scores.
        source_nodes (List[NodeWithScore]): List of source nodes with their scores.
        final_response_metadata (Dict[str, Any]): Metadata associated with the final response.
    """

    nodes: List[NodeWithScore]
    source_nodes: List[NodeWithScore]
    final_response_metadata: Dict[str, Any]

# Define Workflow
class MultiStepQueryEngineWorkflow(Workflow):
    def combine_queries(
        self,
        query_bundle: QueryBundle,
        prev_reasoning: str,
        index_summary: str,
        llm: LLM,
    ) -> QueryBundle:
        """Combine queries using StepDecomposeQueryTransform."""
        transform_metadata = {
            "prev_reasoning": prev_reasoning,
            "index_summary": index_summary,
        }
        return StepDecomposeQueryTransform(llm=llm)(
            query_bundle, metadata=transform_metadata
        )

    def default_stop_fn(self, stop_dict: Dict) -> bool:
        """Stop function for multi-step query combiner."""
        query_bundle = cast(QueryBundle, stop_dict.get("query_bundle"))
        if query_bundle is None:
            raise ValueError("Response must be provided to stop function.")

        return "none" in query_bundle.query_str.lower()

    @step(pass_context=True)
    async def query_multistep(
        self, ctx: Context, ev: StartEvent
    ) -> QueryMultiStepEvent:
        """Execute multi-step query process."""
        prev_reasoning = ""
        cur_response = None
        should_stop = False
        cur_steps = 0

        # use response
        final_response_metadata: Dict[str, Any] = {"sub_qa": []}

        text_chunks = []
        source_nodes = []

        query = ev.get("query")
        await ctx.set("query", ev.get("query"))

        llm = Settings.llm
        stop_fn = self.default_stop_fn

        num_steps = ev.get("num_steps")
        query_engine = ev.get("query_engine")
        index_summary = ev.get("index_summary")

        while not should_stop:
            if num_steps is not None and cur_steps >= num_steps:
                should_stop = True
                break
            elif should_stop:
                break

            updated_query_bundle = self.combine_queries(
                QueryBundle(query_str=query),
                prev_reasoning,
                index_summary,
                llm,
            )

            print(
                f"Created query for the step - {cur_steps} is: {updated_query_bundle}"
            )

            stop_dict = {"query_bundle": updated_query_bundle}
            if stop_fn(stop_dict):
                should_stop = True
                break

            cur_response = query_engine.query(updated_query_bundle)

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Answer: {cur_response!s}"
            )
            text_chunks.append(cur_qa_text)
            for source_node in cur_response.source_nodes:
                source_nodes.append(source_node)
            # update metadata
            final_response_metadata["sub_qa"].append(
                (updated_query_bundle.query_str, cur_response)
            )

            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n" f"- {cur_response!s}\n"
            )
            cur_steps += 1

        nodes = [
            NodeWithScore(node=TextNode(text=text_chunk))
            for text_chunk in text_chunks
        ]
        return QueryMultiStepEvent(
            nodes=nodes,
            source_nodes=source_nodes,
            final_response_metadata=final_response_metadata,
        )

    @step(pass_context=True)
    async def synthesize(
        self, ctx: Context, ev: QueryMultiStepEvent
    ) -> StopEvent:
        """Synthesize the response."""
        response_synthesizer = get_response_synthesizer()
        query = await ctx.get("query", default=None)
        final_response = await response_synthesizer.asynthesize(
            query=query,
            nodes=ev.nodes,
            additional_source_nodes=ev.source_nodes,
        )
        final_response.metadata = ev.final_response_metadata

        return StopEvent(result=final_response)

# Load data
documents = SimpleDirectoryReader("data/paul_graham").load_data()

# Setup LLM
llm = OpenAI(model="gpt-4")

Settings.llm = llm

# Create index and query engine
index = VectorStoreIndex.from_documents(
    documents=documents,
)

query_engine = index.as_query_engine()

async def main():
    #run the workflow
    w = MultiStepQueryEngineWorkflow(timeout=200)

    # Sets maximum number of steps taken to answer the query.
    num_steps = 3

    # Set summary of the index, useful to create modified query at each step.
    index_summary = "Used to answer questions about the author"

    query = "In which city did the author found his first company, Viaweb?"

    result = await w.run(
        query=query,
        query_engine=query_engine,
        index_summary=index_summary,
        num_steps=num_steps,
    )

    print("> Question: {}".format(query))
    print("Answer: {}".format(result))


    # Display step queries created
    sub_qa = result.metadata["sub_qa"]
    tuples = [(t[0], t[1].response) for t in sub_qa]
    print(f"{tuples}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
# If created query in a step is None, the process will be stopped.


