from fastmcp import FastMCP, Client, Context
from typing import Type
from pydantic import (
    BaseModel,
    Field,
    create_model,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastmcp.client.sampling.handlers.openai import OpenAISamplingHandler

from typing import List, Literal
from rich.console import Console
import os


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

console = Console()


mcp = FastMCP(
    "Dynamic Data Extraction MCP Server",
    sampling_handler=OpenAISamplingHandler(default_model="gpt-4o"),
    sampling_handler_behavior="fallback",
)


@mcp.tool()
async def dynamic_schema_extract(
    user_specification: str, schema_name: str, ctx: Context
) -> List[BaseModel]:
    """Extracts data from the provided text into a dynamically generated schema."""

    class Property(BaseModel):
        """property name and corresponding type"""

        name: str = Field(..., description="must be snake case")
        type: Literal["string", "integer", "boolean"]

    class DynamicSchema(BaseModel):
        """schema for the dynamic data extraction"""

        name: str
        description: str
        properties: list[Property]

    def factory(schema: DynamicSchema) -> Type[BaseModel]:
        type_definition = {
            "string": ("str", ""),
            "integer": ("int", 0),
            "boolean": ("bool", True),
        }
        field_definitions = {
            attribute.name: (type_definition[attribute.type])
            for attribute in schema.properties
        }

        DataModel: type[BaseModel] = create_model(
            schema.name.capitalize(),
            __config__=None,
            __doc__=schema.description,
            __base__=BaseModel,
            __module__=__name__,
            __validators__=None,
            __cls_kwargs__=None,
            __qualname__=None,
            **field_definitions,
        )
        return DataModel

    custom_properties = await ctx.sample(
        temperature=0.7,
        max_tokens=300,
        result_type=DynamicSchema,
        system_prompt=("You are a world class data structure extractor. "),
        messages=f"""provide all necessary properties for a {schema_name} data class which captures all given information
            about the {schema_name}s decribed in the text below. 
            text: {" ".join(user_specification)}
            """,
    )

    schema = custom_properties.result
    if not schema:
        raise ValueError("schema for dynamic data extraction is empty")
    await ctx.info(f"Custom properties extracted: {schema}")

    DataModel: type[BaseModel] = factory(schema)

    results = await ctx.sample(
        result_type=List[DataModel],  # type: ignore[valid-type]
        messages=f"parse the text: `{user_specification}`",
        temperature=0,
    )
    # await ctx.info(f"Data extracted: {results.text}")
    return results.result


async def main():

    handler = OpenAISamplingHandler(default_model="gpt-4o")

    async with Client(mcp, sampling_handler=handler) as client:
        user_specification = (
            "extract a nested Contact with Address schema from the following Text: \n"
            "testi testuser \nteststreet 99\n3456 Testcity\nGermany\ngerman\nmale\n"
        )
        result = await client.call_tool(
            "dynamic_schema_extract",
            {"user_specification": user_specification, "schema_name": "Contact"},
        )
        console.print("LLM Response:", result.structured_content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
