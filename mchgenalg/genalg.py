# -*- coding: utf-8 -*-
"""
Based on Martin Chovanec's work at the Università della Svizzera italiana on Sun Dec 13 14:41:12 2015

@author: Martin Chovanec
"""
import numpy as np


# Глобальный вопрос: откуда приходит Геном. Он формируется в бинарной популяции.

# Создается класс. Где создаются экземпляры? Вообще интерфейс совпадает с тестовым исходником
class GeneticAlgorithm():

    # Инициализация. Аналог интерфейса
    def __init__(self, fitness_function): 
        self.population = None
        self.fitness_function = fitness_function
        self.number_of_pairs = None
        self.mutation_rate = 0.005
        self.selective_pressure = 1.5
        # Если 2-е родителей с одинаковым генотипом, заменить их случайными.
        self.allow_random_parent = True
        # Использовать одноточечный кроссинговер вместо равномерного
        self.single_point_cross_over = True
        
    # 1) Генерация бинарной популяции 
    def generate_binary_population(self, size, genome_length):        
        # тип, по умолчанию, long
        self.population = np.random.randint(0, 2, (size, genome_length))
        self._update_fitness_vector()
        return self.population
    
    # Зачем? Генерация особей, почему логический тип?
    def _generate_individual(self, genome_length):  
        return np.random.randint(0, 2, (genome_length), dtype=bool)
    
    # 2) Функция пригодности. Зачем разбита на 3?
    def get_fitness_vector(self):  
        return self.fitness_vector
    def _update_fitness_vector(self):  # Зачем? Обновление функции пригодности
        self.fitness_vector = [self.get_fitness(genom) for genom in self.population]      
    def get_fitness(self, genome):
        return self.fitness_function(genome)

    # 3) Взять лучший геном. Отбор ли?
    def get_best_genome(self): 
        self.best_genome = np.argmax(self.fitness_vector)
        return self.population[self.best_genome], self.fitness_vector[self.best_genome]

    # 4) Запуск ГА
    def run(self, iterations=1):  
        # На случай ошибок:
        if self.population is None:
            raise RuntimeError("Population has not been generated yet.")
        if not self.number_of_pairs:
            raise RuntimeError("The number of pairs (number_of_pairs) to be generated has not been configured.")
        
        # Итерации, похоже, генерация потомства
        for iteration in range(iterations):
            # Из выбранных родителей
            parent_pairs = self._select_parents(self.number_of_pairs, self._get_parent_probabilities())
            for parent_pair in parent_pairs:
                # Генерация детей
                children = self._generate_children(parent_pair)
                # Процесс мутации
                mutated_children = [self._mutate(child, self.mutation_rate) for child in children]
                # Возможные изменения (в сочетании с поиском табу?)
                for child in mutated_children:
                    child_fitness = self.get_fitness(child)
                    # Самый нелучший. Зачем?
                    worst_genome = np.argmin(self.fitness_vector) 
                    if self.get_fitness_vector()[worst_genome] < child_fitness:
                        self.population[worst_genome] = child
                        self.get_fitness_vector()[worst_genome] = child_fitness

    # 5) Ранжировка родителей в соотвествии с фитнес-вектором.
    def _get_parent_probabilities(self):  
        # 2 − SP + 2* (SP −1)* (rank(i) −1) /(N −1)
        relative_fitness = self.fitness_vector / np.sum(self.fitness_vector)
        ranks_asc = np.argsort(relative_fitness)
        return np.array([2 - self.selective_pressure + 2 * (self.selective_pressure - 1)
                         * (ranks_asc[-i] - 1) / (len(ranks_asc) - 1) for i in range(len(ranks_asc))])

    # 6) Получение популяции детей: скрещивание
    def _generate_children(self, parent_pair):  
        parent1 = parent_pair[0]
        parent2 = parent_pair[1]
        # Для одноточечного кроссинговера
        if self.single_point_cross_over:
            cutAfter = np.random.randint(0, len(self.population[0]) - 1)
            # Возвращает массив, состоящий из 2-х
            return (np.concatenate((parent1[0:cutAfter + 1], parent2[cutAfter + 1:])),
                    np.concatenate((parent2[0:cutAfter + 1], parent1[cutAfter + 1:])))
        else:
            # Универсальный кроссинговер:
            # случайно сгенерированные значения, меньшие чем prob =>, берут бит от первого родителя до первого потомка,
            # > = 0.5 взять от второго родителя до первого потомка
            threshold = 0.5
            prob = np.random.rand(len(parent1))
            children = np.ndarray((2, len(parent1)), dtype=bool)
            mask1 = prob < threshold
            mask2 = np.invert(mask1)
            children[0, mask1] = parent1[mask1]
            children[0, mask2] = parent2[mask2]
            children[1, mask1] = parent2[mask1]
            children[1, mask2] = parent1[mask2]

            return tuple(children)
        

    # 7) Отбор родителей исходя из функции пригодности
    def _select_parents(self, number_of_pairs, parent_probabilities):  
        """
        Create an array of pairs (array of parents)
        :param number_of_pairs:
        :type number_of_pairs: int
        :param parent_probabilities: Vector defining how probably will each genome be selected as a parent
        :type parent_probabilities:
        :return:
        :rtype:
        """
        parent_pairs = []
        # Получение количества пар
        for pair in range(number_of_pairs):
            parents_idx = []
            parents = []
            
            while len(parents_idx) != 2:
                rnd = np.random.rand()
                
                for i in range(len(parent_probabilities)): # Почему длина? parent_probabilities - это 1мер. массив
                    p = parent_probabilities[i]
                    if rnd < p:
                        parents_idx.append(i) # Формируется массив в соот с пригодностью
                        parents.append(self.population[i].copy()) # Родители соотносятся в соот с parents_idx
                        
                        # Выход из внут цикла по достижении 2(цели)
                        if (len(parents_idx) == 2):
                            break
                            
                        # Нормализация возможностей, чтобы выбрать другого родителя для скрещивания
                        parent_probabilities += p / (len(parent_probabilities) - 1)
                        parent_probabilities[i] = 0
                        
                        # Мы вернем возможность в конце цикла while
                        firstParentProbability = p
                        break
                        
                    else:
                        rnd -= p
                        
            # Вероятность первого родителя была установлена на 0 во время поиска скрещивания
            # Установите его обратно:
            parent_probabilities -= firstParentProbability / (len(parent_probabilities) - 1)
            parent_probabilities[parents_idx[0]] = firstParentProbability
            
            if self.allow_random_parent and np.all(parents[0] == parents[1]):
                parents[0] = self._generate_individual(len(parents[0]))
                parents[1] = self._generate_individual(len(parents[1]))
            parent_pairs.append(parents)

            # With this solution, it may happen that we have many pairs of the same pair :(
        return parent_pairs

    def _mutate(self, genome, mutation_rate):  # Оператор мутации. NumPy
        rnd = np.random.rand(len(genome))
        mutate_at = rnd < mutation_rate
        genome[mutate_at] = np.invert(genome[mutate_at])
        return genome

