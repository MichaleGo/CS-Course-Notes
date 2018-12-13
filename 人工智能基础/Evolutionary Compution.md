## 第七章 进化计算
#### Outline
---
1. [Part Ⅰ: Biological Inspiration to Search](#1)
2. [Part Ⅱ:DNA Computing](#2)
3. [Part Ⅲ:What is Evolutionary Computation (EC)](#3)
4. [Part Ⅳ:Evolutionary Algorithms (EA)](#4)
    * Genetic Algorithm(GA遗传算法)
    * Evolutionary Programming(EP进化策略)
    * Evolutionary Strategies(ES进化规划)
    * Genetic Programming (GP遗传规划)

   
---
<span id = "1"></span>  
#### Part Ⅰ: Biological Inspiration to Search
1. Major Agents of Genetic Change in Individuals:
    * **Recombination**  
    * **Mutation**
 
2. 进化机制可以分为 **自然选择**、**重组**、**突变** 三种类型

3. **Evolutionary Computation**:
    >Adoption of the evolutionary paradigm to computation and other problems can help us find optimal solutions
    
<span id = "2"></span>  
#### Part Ⅱ : DNA Computing
1. DNA acts as a massive memory, but complementary bases react with each other can be used to compute things.

2. Uniqueness of DNA:
    * Extremely dense information storage.
        >The 1 gram of DNA can hold about 1x1014 MB of data.

    * Enormous parallelism.
        >contain trillions of strands. 3X10^14 
    * Extraordinary energy efficiency.
        >2 x 10^19 operations per joule.

3. [DNA计算](https://wenku.baidu.com/view/8bd3eed85022aaea998f0f62.html)

<span id = "3"></span>
#### Part Ⅲ: What is Evolutionary Computation?
1. The Metaphor between **evolution** and **search**
    >Environment -> Problem
    >Individual -> Candidate Solution
    >Fitness -> Quality(Value or Cost）
    
2. 进化机制可以分为 **自然选择**、**重组**、**突变** 三种类型

3. The Evolutionary Cycle
![d69603a4284825f5f64212fd4ebbe950.png](en-resource://database/906:1)


4. 典型进化算法
    确定问题的表达方式，基因型还是表现型，编码方式
![5d40467976f445593fd1310067f485dd.png](en-resource://database/908:1)


5. 五个要素factor
        >**Representation** Genotypic vs. Phenotypicv
        >**Fitness Evaluation**
        >**Genetic Operations**
        > - Recombination    
        > - Mutationv
        > **Selection Strategies**
        > - Parent Selection    
        > - Survivor Selection初始化终止方案
        > **Initialization\Termination Schemes**

6. 突变exploration和选择exploitation的平衡
    >选择太多会导致局部最优解
    >突变太多会导致不收敛
7. 如何设计一个进化算法？
    >Step1. Design a representation
    >Step2. Design a way of mapping a genotype to a phenotype (not necessarily)
    >Step3. Design a fitness function
    >Step4. Design suitable genetic operators: mutation and/or recombination
    >Step5. Decide how to select parents and survivors
    >Step6. Decide how to initialise a population and when to stop the algorithm
8. [Example: 8 Queen](https://wenku.baidu.com/view/95de3f27b8f67c1cfbd6b801.html)

<span id = "4"></span>
#### Part Ⅳ : Evolutionary Algorithms
1. Genetic algorithms
2. Evolutionary Programming
    * 主要特点
        >表现型观点
        >只存在突变不存在重组
        >进化发生在个体上
        >先产生新的群体，然后在新旧两个群体上平等选择
    * 理论基础
        >One point in the search space stands for a species, not for an individual and there can be no crossover between species

    * 选择算子
        >Parent selection(一个父代个体通过突变产生一个子代个体)
        >Survivor selection（随机型q-竞争法）
    * [Example application: evolving checkers players]()


3. Evolutionary Strategies
    * 进化策略主要特点
        > applied to numerical optimazation
        > fast, good optimizer for real-valued optimazation, relatively much throry
        >self-adaptation of (mutation) parameters standard
        >自然选择按照确定的方式执行，有别于遗传算法和进化规划中的随机选择方式
    * 进化策略的不同形式
        * (1+1)-ES, only mutation
        * (u + 1)-ES, mutation and recombination
        * (u + λ)-ES, mutation and recombination
    * 进化策略中的遗传操作
        * 重组
            *  离散重组
            *  中值重组
            *  混杂重组
        * 变异
            * 简单突变算子
            * 二元突变算子
            * 三元突变算子![a0b40403dbd8d354b4c51b1786c410d3.png](en-resource://database/912:1)
    * 选择
        > 严格按照适应度大小选择
    * 算法流程
        > Step1  确定问题的表达方式
        > Step2  随机生成初始种群，计算其中每个个体的适应度
        > Step3  用以下操作生成新群体，实现进化：
            > Step3-1. 选择某种重组方式进行个体重组   
            > Step3-2. 选择某种突变方式对重组后的个体进行突变   
            > Step3-3. 计算新个体适应度   Step3-3. 选择适应度最高的前若干个优良个体组成下一代群体
            > Step4  反复执行Step3，直到终止条件满足，从群体中选择最优个体作为最优解。这里，终止条件与进化规划算法的终止条件类似。
            
