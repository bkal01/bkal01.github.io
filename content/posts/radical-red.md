---
title: "Can Coding Agents Play Pokemon?"
description: "A benchmark for coding agents based on difficult Pokemon battles, where agents build teams, learn from losses, and adapt their strategy."
date: 2026-09-02
draft: false
math: true
cover:
  image: "/images/radical-red/fable-team.png"
  hidden: true
---

<div class="tldr">
    <strong>TL;DR</strong>
    <br>
    I made a new benchmark for coding agents based on 21 difficult Pokemon battles from the fanmade ROM hack Radical Red. Agents need to inspect data in a sandbox to build a team and battle an opponent. As they uncover more information about their opponent, they can update their team under a limited episode budget.
    <br><br>
    In an initial eval, GPT 5.6 Luna wins ~83% of its battles within 10 episodes. Pretty strong, but we have levers to scale task difficulty up for future evaluations.
    <br><br>
    Code is <a href="https://github.com/bkal01/claude-radical-red">here</a> if you want to run some evals yourself or contribute!
</div>

# Introduction

Recently, Anthropic announced that their new model, Claude Fable 5, [was able to beat Pokemon FireRed using vision only](https://www.youtube.com/watch?v=Ty_50J84fMY). Impressive! Like every other benchmark we've set for AI, Pokemon FireRed too has been saturated. So what now?

We build a harder benchmark of course!

Let's take a closer look at the team Claude used to beat the game:

![Screenshot from the launch video taken by X user @PhDenLogica. Charizard is level 76 while the rest of the Pokemon in the party are level 25 or lower.](/images/radical-red/fable-team.png)

It seems like Claude adopted a primitive strategy of over-leveling its starter Pokemon rather than building a team with proper type coverage and supporting Pokemon. This is a passable strategy for Pokemon FireRed. Anyone who's played any of the mainline Pokemon games recently can tell you that they're quite easy, which is why players typically self-impose challenges such as Nuzlocke/Mono-Type.

The FireRed ROM hack [Pokemon Radical Red](https://www.pokecommunity.com/threads/pok%C3%A9mon-radical-red-version-4-1-released-gen-9-dlc-pokemon-character-customization-now-available.437688/) offers a more challenging experience for players. It contains all Pokemon/moves/items up to Generation 9 and difficult boss battles. These boss battles often have cohesive teams with lots of type coverage, useful items (e.g. Focus Sash and White Herb), and max EVs. You cannot use items during boss battles and it's impossible to over-level for them due to level caps/adaptive boss levels.

Radical Red is a perfect candidate for the next iteration of Pokemon benchmarks for LLMs. In order to succeed, models will need to understand a larger pool of Pokemon/moves/items/mechanics, learn from their losses, and develop cohesive teams and strategies.

# Benchmark Details

For the sake of time/money, rather than having agents play the full game, it's better to focus on just the battles rather than navigating around the map. Apart from battling, even Radical Red is not difficult, it is just a matter of going where the characters tell you to.

## Tasks

Radical Red features *boss battles*, which are much more difficult than standard battles. These involve:

- set mode instead of shift mode, meaning you don't know what Pokemon is coming next after you knock one out
- a level cap/adaptive levels, so you can't overlevel your Pokemon
- competent teams with coverage moves and useful items
- maxed out EV spreads

I extracted 21 boss battles from Radical Red into "Tasks" for the benchmark. Each task has a level cap and a carefully curated list of Pokemon/items/TMs. Additionally, each task has a restriction on the maximum allowed party size, which is capped at the number of Pokemon the enemy trainer has. These tasks range in difficulty, from somewhat easy trainers encountered on routes to gym leaders and even the leader of Team Rocket.

We'll specifically evaluate *coding agents* on these tasks. The idea is that we drop an agent into a sandbox with all the game data present as JSON files. It can grep through those files to understand what Pokemon are available for its team. Then, it can submit commands to an MCP server that holds the battle state and processes the agent's actions. Note that this isn't a typical coding agent eval like TerminalBench/FrontierSWE: the SWE capabilities required here are not that hard, we're just using coding agents as a convenient harness.

## Available Actions

The MCP server supplied to the agent accepts a few different commands:

- `apply_team`: takes in a full team configuration and constructs the corresponding party in game data
    - the agent has full control of the Pokemon in its team, their EV spreads, items, Natures, Abilities, and moves
- `observe`: snapshot of the current state of the battle
- `lead`: choose the Pokemon to start the battle with
- `action`: 1 of 3 possible actions during a battle
    - `FIGHT`: use a move that the currently active Pokemon knows
    - `SWITCH`: switch to another Pokemon in the Party
    - `SEND`: used only when the active Pokemon is KO'd or in other [niche](https://bulbapedia.bulbagarden.net/wiki/Emergency_Exit_(Ability)) [scenarios](https://bulbapedia.bulbagarden.net/wiki/U-turn_(move))
- `reset`: restarts the battle from the beginning

## Evaluating Agents

We'll adopt some [terminology from the Harbor framework](https://www.harborframework.com/docs/core-concepts) plus some of our own:

- as mentioned above, a *Task* is a specific battle we evaluate an agent on
- an *Episode* is a single battle attempt: the agent calls `apply_team`, then `lead`, then repeatedly calls `action`
    - `reset` ends the current episode and starts a new one
- a *Trial* is a full battle attempt by the agent, including any planning/analysis done by the agent and one or more episodes
    - trials have an episode budget, basically the number of attempts an agent gets at the battle

We judge agents based on:
1. whether a trial is successful or not
2. how many episodes were needed to reach success

To be precise, we'll look at the following metric:

$$
\text{win}_{\leq E} = \frac{1}{N}\sum_{i}^{N} \mathbf{1}[\text{won}_i \land \{\text{episodes}_i \leq E\}]
$$

which is the fraction of $N$ trials won within $E$ episodes. If an agent needs fewer episodes to succeed in a trial, then it either picked a good starting team or optimized its team very efficiently, both of which should be rewarded.

## Scaling Difficulty

The tasks in this benchmark already naturally fall into different difficulty tiers. Some are easily one-shottable while others might take dozens of episodes. Of course agents are ever-improving and we'll probably reach a point soon where all of these task are easily solved. Luckily we have knobs that we can turn to easily scale up the difficulty of a task to arbitrary levels. If a task is too easy, we can:

- scale the level cap down by 5-10%
- decrease the maximum allowed party size
- exclude specific Pokemon/items/moves

All of these are as simple as a config change + running a script in the codebase!

# Results

![Cumulative benchmark performance of GPT-5.6 Luna (high) in Codex on 21 tasks. Each task was evaluated in 5 independent trials with an episode budget of 10. The plotted metric shows the fraction of all trials that succeeded within $E$ episodes. The agent wins 34.3% of trials at $E=1$, all the way up to 82.9% trials at $E=10$](/images/radical-red/gpt-5-6-luna-results.png)

We ran 5 independent trials for each of the 21 tasks, with each trial having an episode budget of 10. The model/harness used was GPT-5.6 Luna with high reasoning effort inside of Codex.

34.3% of trials were won in the first episode: these are almost entirely the "easy" tasks, usually random trainers on a route that are just slightly stronger than average. Most well-built teams at the level cap can snag a win pretty easily here. Within 10 episodes, the success rate jumps to 82.9% of the trials. The trials that were unable to be won within 10 episodes belonged to Radical Red's harder battles, which are gym leaders and Team Rocket bosses.

By the curve, we can also see some diminishing returns: there's a large jump in performance within 2 episodes, likely because the agent gets most of the information it needs in the first episode and is easily able to build a winning team from that. As the number of episodes increases, success rate still climbs, but more slowly. The agent is able to adapt its strategy within its budget to capture more wins but it's not enough for the really difficult battles.

## Task Analysis

We'll take a look at 3 different tasks of varying difficulties to see how the agent chose its team and adapted its strategy.

### Lass Anne, Viridian Forest

![Lass Anne's team, which has 3 different threats that operate in completely different ways. Stufful is defensive against physical attacks, Clefairy can luck into some high damage with Metronome, and Audino can stall the agent out while continuously healing.](/images/radical-red/lass-anne-viridian-forest.png)

`lass-anne-viridian-forest` is a battle from one of the earliest routes in the game. The core challenge behind the task is an Audino that stalls you out with Swagger/Protect/Wish/Yawn, along with a Clefairy that gets a free 30% damage boost with Life Orb + Magic Guard and spams Metronome. In my experience, Lass Anne is not really difficult, but it is designed to ragebait you. A long, drawn-out battle that has an RNG component can tilt a player and cause unforced errors. Luckily agents don't get mad as easily as humans!

For this task, the agent is able to win 5/5 Trials on the very first episode. In each Trial, it follows the same general search strategy to build its initial team: scanning over the available Pokemon (either via Node.js scripts or `jq` queries), sorting them by stat totals, inspecting the strongest legal moves of high-statted Pokemon, then building a three-Pokemon team with good type coverage. It landed on a different strategy every time:

- Trial 1: Vivillon, Mienfoo, Houndour
- Trial 2: Houndour, Mienfoo, Roselia
- Trial 3: Frosmoth, Roselia, Galarian Ponyta
- Trial 4: Flaaffy, Oshawott, Mankey
- Trial 5: Marill, Houndour, Staravia

Watch the agent beat Lass Anne:
<video src="/videos/radical-red/lass-anne-viridian-forest.mp4" controls width="100%"></video>

### Lt. Surge, Vermilion City

![Lt. Surge's team. Note that every Pokemon has an ability to deal with Ground type counters, whether its Ice/Water/Grass type coverage moves, the Shuca Berry, or the Levitate ability. Lt. Surge also has a Mega Evolution in Mega Manectric, which gives double Intimidate and an extremely fast special sweeper w/Charge Beam boosting its SPATK. Pincurchin sets up the Electric Terrain, which boosts Electric type moves by 50%, and Bellibolt is an annoying stall Pokemon that heals itself up.](/images/radical-red/lt-surge-vermillion-city.png)

#### Enemy Team Description

`lt-surge-vermillion-city` is the Electric type gym leader in Radical Red. Defensively, Electric is a pretty good type as it only has one weakness, Ground. However, relying on Ground type moves is a poor strategy for Lt. Surge, as every single Pokemon has a way to deal with them: multiple Pokemon have Ice/Water/Grass type coverage moves, Bellibolt holds the Shuca Berry which reduces the damage of Ground type moves, and Vikavolt has Levitate which makes it immune to Ground type moves.

The gym leader has two additional advantages that make this battle tough: Pincurchin sets up the Electric Terrain which boosts Electric type moves by 50%, and he also has access to a Mega Evolution, which the agent does not.

#### Agent Performance

The agent won 1/5 trials for this task. For the trial that it won, it actually won in the first episode, which is pretty wild! It landed on a really solid initial team with Mamoswine for Electric immunity and Annihilape with Defiant for a great counter to Intimidate on Manectric. It demonstrated good in-battle strategy by switching into Mamoswine to dodge key Electric type attacks and using Fire Punch to beat Vikavolt.

Let's look at a trial where the agent failed though, so we can analyze how it tried to adapt its team and strategy based on the information it uncovered. 

The agent started the trial off with a team of Lucario, Gardevoir, Excadrill, Mamoswine, and Luxray. Great coverage with incidentally two Electric immunities in Excadrill and Mamoswine. As the agent battled, it quickly learned that Ground type Pokemon alone are not sufficient. It discovered that Vikavolt has Levitate, Bellibolt has Water coverage, and Pawmot has the Focus Sash.

As the agent progressed through its episodes, it realized that it was struggling against Bellibolt's strong Water type move. It chose to counter this with Pokemon that have Storm Drain (Gastrodon and Cradily), which makes the user immune to Water type moves and boosts SPATK instead of taking damage. The agent used these as targeted counters to absorb Water type attacks that would otherwise KO its Ground type Pokemon.

Here's a table showcasing all the adaptations made by the agent:

| Observation | Adaptation | Team Modification |
| --- | --- | --- |
| Vikavolt's Levitate dodges Ground type attacks | Rock/Flying is a good matchup | Archeops w/Rock Tomb |
| Bellibolt uses Muddy Water | Storm Drain can counter this | Gastrodon/Cradily |
| Pawmot's Focus Sash lets it survive attacks | Stealth Rock provides chip damage | Graveler |
| Pawmot's priority Mach Punch sweeps weakened teams | Burning Pawmot and changing its ability can reduce damage | Cofagrigus w/Mummy, Will-O-Wisp, Hex |


The agent clearly identified problematic components of Surge's team and made targeted changes to counter them. In the final episode, the agent was able to get the last Pokemon, Pawmot, down to 1 HP, but was unable to win.

The agent is quite strong at in-battle adaptation, like using super effective moves and switching to dodge attacks and get favorable type matchups. It also learns the key interactions of the battle quickly. Its general strategy is to build a balanced team initially, gather information in the first few episodes, and then iteratively patch failures that come up as it goes. Its solutions to these failures are quite creative in my opinion (I did not consider Storm Drain or Cofagrigus when making my own team for Surge).

The problem with the agent's approach is that it is very *local*. The agent repeatedly follows the pattern of encountering a failure and making one or two team changes to account for it. But each answer added to the team sometimes weakened another matchup. As mentioned earlier the agent added Archeops to beat Vikavolt. But in the final episode, it removed Archeops and as a result could not easily beat Vikavolt.

A different strategy would be to take a step back and consider a *globally optimal* solution. When I battled Lt. Surge myself, once I learned the full enemy team, I considered the full matchup and thought about what I wanted my overarching strategy to be. I landed on the following core which is a pretty good starting point globally and only requires a few local updates to produce a winning solution:

- Hypno can change Electric Terrain to Psychic Terrain and is bulky enough to survive several hits while dealing damage and putting Pokemon to sleep
- Torkoal sets up the sun, can tank hits, and deals massive damage
- Camerupt benefits from Torkoal's sun, is immune to Electric, and can survive Muddy Water in the sun

The agent did occasionally make large team changes, but the vast majority of its updates were small local changes.

Watch the agent try to beat Lt. Surge:
<video src="/videos/radical-red/lt-surge-vermillion-city.mp4" controls width="100%"></video>

### Pokemon Tower Ghost

![The ghost of Pokemon Tower is an Alolan Marowak (Fire/Ghost) that is two levels higher than the level cap, has a custom ability that lets it hit everything for at least neutral damage, and literally illegal stats.](/images/radical-red/ghost-pokemon-tower.png#center)

#### Enemy Description

In mainline Pokemon games, the encounter with the ghost of Marowak in Pokemon tower is just a regular battle against a regular Marowak. In Radical Red, the battle is instead against Alolan Marowak (a Fire/Ghost type) and every difficulty knob is turned up to 10. Every single relevant stat has 252 EVs, which is not even a legal setup in Pokemon. The custom ability Bone Zone ensures Bonemerang/Shadow Bone cannot be resisted and also lets it hit Flying type Pokemon and bypass Levitate. Thick Club doubles Marowak's attack, and Marowak also gets Fire/Thunder coverage. To top it all off, when the battle starts, the Marowak gets a +1 boost to every stat.

Although this is a deliberately overpowered Pokemon, there are ways to beat it. With a 6 Pokemon team, you can use Prankster + Toxic on the first turn to ensure Marowak gets badly poisoned, and then spam Sucker Punches with your remaining 5 Pokemon to KO the Marowak as it gets weakened every turn.

#### Agent Performance

The agent won 3/5 trials for this task. In the three won trials, it adopted the following strategies:

1. Slowbro, bulky with Yawn to put Marowak to sleep and then Scald to chip away
2. Lickilicky, bulky with Toxic, Protect, and Rest to stall out while the poison stacks up
3. Primarina, which can one-shot Marowak with Hydro Pump

Pretty cool diversity in approaches! We see sleep, stall, and "sweeping" all adopted by the agent. The approach that I found is using Tyranitar w/Black Glasses + Crunch + Sand Stream, which is able to KO Marowak in two hits due to some weird behavior with the enemy AI (it does not use its super effective moves for some reason).

Because the team size for this task is 1, Pokemon updates are inherently "global", while updating EVs, items, etc. is local. In this task, the agent mostly made global updates, constantly changing Pokemon to vary its strategy.

Watch the agent beat the Pokemon Tower Ghost:
<video src="/videos/radical-red/ghost-pokemon-tower.mp4" controls width="100%"></video>

## Agent Tendencies

This benchmark also gives us a unique opportunity to explore agent tendencies. We know already that the agent will pick generally strong Pokemon (based on stats, typing, etc.). What's more interesting is to see if there are any other biases the agent has. Specifically on two fronts:

1. Pokemon Generations: the last Pokemon generation came out in 2022, so it's fair to assume that all Pokemon have been included in the training data for modern LLMs. The different Pokemon generations have varying levels of representation in the training data which is somewhat related to how long they've been around + their relative popularity. Given this, do agents have a tendency to pick certain generations over others?

2. Team Updates: we already know the agent biases towards local team updates. But what do these updates look like? Is the agent changing entire Pokemon, or is it changing stat spreads and moves?

### Pokemon Generations

There are two forces pulling in opposite directions when it comes to influencing agent choices in Pokemon generations. On one hand, earlier generations are likely overrepresented in an LLM's training data. They've been around for longer and are very popular among casual players and competitive players alike. An agent is probably more familiar with Dragonite than it is with Urshifu. On the other hand, Pokemon generally experiences "power creep", where newer Pokemon generally have more overloaded kits than older ones. Of course this is not a blanket rule, but if the agent is analyzing data objectively then I would expect it to bias towards newer generations.

To examine agent behavior here, I created a dummy task with a level cap of 100 and all Pokemon available (besides Megas, Legendaries, and Mythical Pokemon). I then had the agent select a team of 6 Pokemon 100 times independently for an unknown task.

![Observed counts of Pokemon / Expected counts of Pokemon, per generation. Gen 4 and Gen 9 are very popular, Gen 5 is as expected, and the remaining generations are all underrepresented.](/images/radical-red/generation-bias.png)

The figure above shows the ratio between the observed and expected counts of Pokemon per generation. We use a ratio rather than raw observed counts because the number of available Pokemon varies per generation. As we can see, Gen 4 and Gen 9 have extremely high ratios, indicating that the agent selected them much more often than expected. Gen 5 is chosen at an expected amount, and all other generations are underselected.

Here's a table showing the top 3 most frequently chosen Pokemon per generation:

| Generation | Frequency | Top 3 Pokemon + Frequency |
| --- | --- | --- |
| 1 | 77 | Dragonite 49, Starmie 10, Gyarados 8 |
| 2 | 37 | Scizor 14, Azumarill 12, Tyranitar 11 |
| 3 | 56 | Metagross 46, Milotic 6, Salamence 1 |
| 4 | 122 | Garchomp 71, Rotom-Wash 33, Togekiss 13 |
| 5 | 100 |  Volcarona 78, Ferrothorn 15, Excadrill 7 |
| 6 | 17 | Greninja 14, Aegislash 2, Florges 1 |
| 7 | 6 | Primarina 6 |
| 8 | 19 | Dragapult 18, Rillaboom 1 |
| 9 | 166 | Kingambit 47, Great Tusk 34, Flutter Mane 32 |


Gen 4 and Gen 9 are dominated by very strong generic picks. Pokemon like Garchomp, Rotom-Wash, and Kingambit have been competitive mainstays. The agent also identifies the Paradox Pokemon from Gen 9 (Great Tusk, Flutter Mane, etc.) as strong picks. It seems like among other generations, the agent finds fewer strong, generically applicable Pokemon.

So not necessarily a "generation bias", but moreso the bias towards strong generic Pokemon naturally narrows down the available Pokemon to a shortlist that is overloaded in Gen 4/9.

### Team Updates

Optimizing a team is a very difficult task. You can put a Pokemon on your team, but it might not work until you give it exactly the right EV spread, moves, items, and ability. The most common thing pro players do is to run damage calculations and tune EVs/Natures to survive key threats, secure KOs, and outspeed specific Pokemon. We want to know if agents approach the problem of teambuilding the same way humans do and if they have the patience to stick with a Pokemon that's not working immediately and tweak it to get a better result.

Over the 105 trials, there are several instances of the agent tuning existing Pokemon on its team for a specific goal. Here are some examples:
- In the `whitney-route-11` task, the agent changed its Skuntank's Nature from Rash (+SPATK/-SPDEF) to Calm (+SPDEF/-ATK) in order to take more hits
- In the `giovanni-silph-co` task, the agent gave Infernape an EV spread of 4 HP/128 ATK/124 SPATK/252 SPE so that it could deal both physical and special damage
- In the `erika-celadon-city` task, the agent updated Arcanine from 4 HP/252 ATK/252 SPE to 252 HP/252 ATK/4 SPE because it determined that it could not outspeed an enemy Meowscarada and so instead it tried to out-bulk it

So the agent does have the capability to make principled, fine-grained changes to its team. But more often it just chooses to replace a Pokemon that's not working with another one entirely. This is probably related to the limited episode budget. Making a principled change is not just a matter of applying the change, you also need to reproduce the exact scenario that comes up to see if the change solves the problem you were facing. This is hard to do when the enemy AI is non-deterministic. So, I suspect the agent thought it would get more value per team update by just fully replacing a failing Pokemon rather than tweaking it. As the episode cap $\rightarrow \infty$, I would assume the agent will try more granular updates.


# Conclusion

This benchmark in particular asks a few interesting questions of coding agents. Can an agent develop a strategy over a long-horizon under partial observability? Can it update that strategy as it gets exposed to new information? And can it manage a limited budget to efficiently explore the vast search space of Pokemon?

To complete v1 of this benchmark, I want to evaluate more models and more coding agents. Different models might have different tendencies, and stronger models might plan and adapt in more interesting ways. It would be interesting to see agents produce scripts for replaying battles offline, or create custom damage calculators that update with battle evidence. There's also more work needed for the benchmark itself. We've covered just about half the game so far, so there are plenty more tasks that can be added. These tasks come with more battle mechanics (Mega Evolution, Doubles, etc.) which make the game more difficult for the agent.

On a final note, I think games (in this case Pokemon battles) are a really interesting way of evaluating coding agents. Or rather, coding agents are a great harness to evaluate models on certain kinds of non-coding tasks. A lot of smart engineering/science has already been done to handle things like context management and tool use. Interacting with the environment is as simple as defining an MCP server. A sandbox lets us give the model access to a large amount of reference data. And a model can reason efficiently through repeatable and modifiable scripts. I think it's better to ship a benchmark that tests the capabilities of models/harnesses that people actually use even if the task itself is unrelated!