# Building a coding agent for index generation

## Structure of the generated code

- We will need the test code to be hand-coded and not touched by the coding agent to avoid reward hacking or manipulation of tests
	- This means that coding agents will need to conform to some sort of structure in order for their generated code to work with a static testing framework
- There should be detailed logs and feedback for LLMs to understand how their code did.
	- Maybe a test on a small example corpus and then can test on building the whole corpus as well
- There should be some way of accessing memory learned from previous experiments and code generation steps. State management over time would be important for real world applications where the index is changing many times over a long period of time
- **Maybe there should be a separate agent for analyzing eval results to pass on to the coding agent as feedback and insights to learn from**

## Proposal 1 - Use existing coding harnesses
- Can use API key with gemini CLI, Codex, and most likely with claude code
- We could run these existing coding harnesses within a folder which builds our preprocessing and index scripts.
- The agent can then call a premade script for testing their code and iterate from there
- We can control its token usage and effort with prompting but this may be hardish
- Can still evolve prompts in some way by putting prompts in a folder and testing with these coding harnesses
### Tradeoffs
- Pros
	- Allows us to continue with experimentation and higher level systems thining for the project without having to think about the implementation of how the coding agent will actually create code
	- Coding harnesses have been built over months by professionals, they will be very good and better than what we could make most likely
- Cons
	- Less customization, we may run into some headaches when we want to change the coding agents performance beyond changing the prompts
	- Not sure how the research community feels about using this kind of thing for open research. They might prefer using custom tools or at least open source frameworks

## Proposal 2 - Create our own coding harness
- From scratch we can design our own coding harness
- This would involve a few parts
	- Designing a context management system - how to represent the repo to the agent? How to store chat history as you go along?
	- How to give it access to terminal and coding tools?
	- Context management to reduce context usage
- Allows for more control and adjustments as we iterate
- Would involve using open source libraries/frameworks like
	- MCP servers
	- Agent2Agent protocol
	- Direct API calls to LLM providers

### Tradeoffs
- Pros
	- Maximum customization
	- If we think the coding loop of the index generation will not be too complex, I think this is best probably
- Cons
	- Added complexity, will take more time
	-  If we are going to have complex engineering cycles where the coding agent can create many files and experiment very widely, we may want to go with prewritten coding harnesses instead


## Proposal 3 - Use open source coding harnesses
- Kind of a middle ground between 1 and 2. Allows for more customization than proposal 1, but also depending on a lot of prewritten patterns and structure
	- [Cline](https://cline.bot/)
	- [OpenCode](https://opencode.ai/)
	- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) (Claude agent SDK is not open source but is a lot more customizable than the gemini CLI and Codex coding harnesses)

### Tradeoffs
- Pros
	- Potentially good balance between proposals 1 and 2
- Cons
	- Potentially steep learning curve