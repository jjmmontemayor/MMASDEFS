#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:02:54 2018

@author: jennifer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:39:32 2018

@author: jennifer
"""

import numpy
import pandas
import random
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from ast import literal_eval
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import data_source

numpy.set_printoptions(threshold=numpy.nan)
numpy.set_printoptions(precision=3)

class MMASDEFS :

    data_instances          = []
    data_classes            = []
    data_classes_instances  = {}
    data_class              = 0
    results                 = []
    best_result             = []         
   
    def __init__(self, iterations, alpha, beta, phi, max_pheromone, min_pheromone, q, f, cr) :

        
        self.loadData()
        
        
        self.ITERATIONS     = iterations
        self.ALPHA          = alpha
        self.BETA           = beta
        self.PHI            = phi
        self.MAX_PHEROMONE  = max_pheromone
        self.MIN_PHEROMONE  = min_pheromone
        self.Q              = q
        self.ANT_COUNT      = self.FEATURE_COUNT
        
        self.F              = f
        self.CR             = cr

        #setup numpy arrays
        self.data_instances = numpy.array(self.data_instances)
        self.data_classes = numpy.array(self.data_classes, dtype=int)
        self.ant_steps = numpy.zeros((self.FEATURE_COUNT), dtype=object)
        self.de_ant_steps = numpy.zeros((self.FEATURE_COUNT), dtype=object)
        self.pheromone_matrix = numpy.zeros((self.FEATURE_COUNT,self.FEATURE_COUNT), dtype=object)
        self.de_pheromone_matrix = numpy.zeros((self.FEATURE_COUNT,self.FEATURE_COUNT), dtype=object)
        self.heuristic_matrix = numpy.zeros((self.FEATURE_COUNT,self.FEATURE_COUNT), dtype=object)
        
        #setup solution object
        self.best_solution = { 0: [], 1: 0, 2: self.FEATURE_COUNT, 3: [], 4: [] }
        self.de_best_solution = { 0: [], 1: 0, 2: self.FEATURE_COUNT, 3: [], 4: [] }
        
        self.de_running = 0
        

    def loadData(self):
        
        self.data_classes, self.data_instances, self.data_classes_instances = data_source.getShuttleData('train')
        self.result_filename = 'shuttle_bmmasdefs.txt'
        self.FEATURE_COUNT = len(self.data_instances[0])
        self.ANT_COUNT = self.FEATURE_COUNT
        
    def evaluateSolution(self, solution, is_best_solution):
        
        test_data_classes,test_data_instances, test_data_classes_instances  = data_source.getShuttleData('test')
        eval_data_instances  = self.data_instances[:, solution[0][0]]

        knn = KNeighborsClassifier()
        knn.fit(eval_data_instances,self.data_classes)
        test_data_pred_classes = knn.predict(test_data_instances[:, solution[0][0]])
        
#        clf = svm.LinearSVC(random_state=0)
#        clf.fit(eval_data_instances,self.data_classes)
#        test_data_pred_classes = clf.predict(test_data_instances[:, solution[0][0]])
        
        
        target_names = []

        for c in test_data_classes_instances.keys() :
            target_names.append('Class ' + str(int(c)))
        
        

        report = classification_report(test_data_classes, test_data_pred_classes, target_names=target_names)
        
        return self.addToResult(report, solution, is_best_solution)

    def addToResult(self,cr, solution, is_best_solution):
        
        # Parse rows
        tmp = []
        
        for row in cr.split("\n"):
            parsed_row = [x for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)
        result = [solution[1], tmp[-1][1], tmp[-1][2], tmp[-1][3], solution[2], numpy.array_str(solution[0])]
        self.results.append(result)
        if(is_best_solution): self.best_result = result
       
    def generateReport(self) :
        numpy.savetxt(self.result_filename, self.results, fmt='%5s', delimiter=',', newline='\n')
        
    def initAnts(self):
        feature_set = random.sample(range(self.FEATURE_COUNT), self.FEATURE_COUNT)

        for row in range(self.ant_steps.shape[0]):
                step = numpy.array([feature_set[row]])
                state = numpy.array([numpy.random.randint(2)])
                self.ant_steps[row] = {0 : step, 1: state, 2: 0.0}
    
    def initDEAnts(self):
        feature_set = random.sample(range(self.FEATURE_COUNT), self.FEATURE_COUNT)

        for row in range(self.de_ant_steps.shape[0]):
                step = numpy.array([feature_set[row]])
                state = numpy.array([numpy.random.randint(2)])
                self.de_ant_steps[row] = {0 : step, 1: state, 2: 0.0}
                
    def initPheromoneMatrix(self) :    
        for row in range(self.pheromone_matrix.shape[0]): 
            for column in range(self.pheromone_matrix.shape[1]):
                    #self.pheromone_matrix[row,column] = numpy.array([[self.MIN_PHEROMONE,self.MIN_PHEROMONE],[self.MIN_PHEROMONE,self.MIN_PHEROMONE]])
                    self.pheromone_matrix[row,column] = numpy.array([[self.MAX_PHEROMONE,self.MAX_PHEROMONE],[self.MAX_PHEROMONE,self.MAX_PHEROMONE]])
                    
    def initDEPheromoneMatrix(self) :    
        for row in range(self.de_pheromone_matrix.shape[0]): 
            for column in range(self.de_pheromone_matrix.shape[1]):
                    #self.pheromone_matrix[row,column] = numpy.array([[self.MIN_PHEROMONE,self.MIN_PHEROMONE],[self.MIN_PHEROMONE,self.MIN_PHEROMONE]])
                    self.de_pheromone_matrix[row,column] = numpy.array([[self.MAX_PHEROMONE,self.MAX_PHEROMONE],[self.MAX_PHEROMONE,self.MAX_PHEROMONE]])


    def initHeuristicMatrix(self):
        for row in range(self.heuristic_matrix.shape[0]):
            for column in range(self.heuristic_matrix.shape[1]):
                    self.heuristic_matrix[row,column] = numpy.array([ [self.getHeuristicValue(0,row,0,column),self.getHeuristicValue(0,row,1,column)],
                                                                  [self.getHeuristicValue(1,row,0,column),self.getHeuristicValue(1,row,1,column)] ])
       
    def evaporatePheromones(self):
        for row in range(self.pheromone_matrix.shape[0]): 
            for column in range(self.pheromone_matrix.shape[1]):
                    self.pheromone_matrix[row,column][0,0] = self.pheromone_matrix[row,column][0,0] - self.PHI
                    self.pheromone_matrix[row,column][0,1] = self.pheromone_matrix[row,column][0,1] - self.PHI
                    self.pheromone_matrix[row,column][1,0] = self.pheromone_matrix[row,column][1,0] - self.PHI
                    self.pheromone_matrix[row,column][1,1] = self.pheromone_matrix[row,column][1,1] - self.PHI


    def getHeuristicValue(self,state_from, node_from, state_to, node_to) :

        total_f_score = 0
        node_to_f_score = 0
        #print('computing heuristic value for sub-paths',node_from, state_from, node_to, state_to)

        for f in range(0, self.FEATURE_COUNT):
            f_score = self.getFScore(f)
            total_f_score += f_score
            if f == node_to :
                node_to_f_score = f_score

        #pearson = pearsonr(self.data_instances[:,node_from], self.data_instances[:,node_to])


        if (state_from == 0 and state_to == 0) or (state_from == 1 and state_to == 0) :
            return (state_from / self.FEATURE_COUNT) * total_f_score
            #return pearson[0]
        elif (state_from == 0 and state_to == 1) or (state_from == 1 and state_to == 1) :
            return node_to_f_score
            #return 1 - pearson[0]

    def getFScore(self, feature) :
        
        n = d = 0
        
        feature_mean_all_classes = numpy.mean(a.data_instances[:,feature])
        
        for k in self.data_classes_instances.keys():
            class_k_data = self.data_classes_instances[k]
            #class_k_data_n_samples = len(class_k_data)
            feature_k_in_class_k = class_k_data[:,feature]
            feature_k_in_class_k_samples = len(feature_k_in_class_k)
            feature_mean_in_class_k = numpy.mean(feature_k_in_class_k)
            n += numpy.square(feature_mean_all_classes - feature_mean_in_class_k)
        
            dd = 0
            
            for j in range(0,feature_k_in_class_k_samples):
                dd += numpy.square(feature_k_in_class_k[j] - feature_mean_in_class_k)
            
            d += (1 / (feature_k_in_class_k_samples - 1)) * dd
        
        print('f-score', n, d, n/d)              
        return n/d

    
    
    def getSolutionQuality(self, classes, new_instances) :
    
        knn = KNeighborsClassifier()
        knn.fit(new_instances, classes)
        return knn.score(new_instances,classes)
    
#        clf = svm.LinearSVC(random_state=0)
#        clf.fit(new_instances,classes)
#        return clf.score(new_instances,classes)
    
    def getIterationBestSolution(self, ant_steps) :
        score_vector = []
        reduction_vector = []
        current_best_solution = { 0: [], 1: 0, 2: self.FEATURE_COUNT, 3: [], 4: [] }
        
        for a in range(0, self.ANT_COUNT) :
            new_features        = numpy.nonzero(ant_steps[a][1])
            new_data_instances  = self.data_instances[:, new_features[0]]
            score_quality       = self.getSolutionQuality(self.data_classes, new_data_instances)
            print("Getting iteration best solution: new features ", new_features," score quality ", score_quality)
            #selection
            if self.de_running == 1 and score_quality < self.ant_steps[a][2] :
                pheromones = self.getPheromonePath(self.ant_steps[a])
                self.updateDEPheromonePath(self.ant_steps[a], pheromones)
                score_quality = self.ant_steps[a][2]
                new_features = numpy.nonzero(self.ant_steps[a][1])
                ant_steps[a] = self.ant_steps[a]
                
            ant_steps[a][2]     = score_quality
            score_vector.append(score_quality)
            reduction_vector.append((self.FEATURE_COUNT - len(new_features[0])) / self.FEATURE_COUNT)
        
        print('Getting iteration best solution: score vector', score_vector)
        print('Getting iteration best solution: reduction vector', reduction_vector)
    
        ants_max_score_indices = numpy.where(score_vector == numpy.max(score_vector))
        #ants_max_reduction = numpy.take(reduction_vector, ants_max_score_indices[0])
        #ants_max_reduction_indices = numpy.where(ants_max_reduction == numpy.max(ants_max_reduction))
        

        max_ant = ants_max_score_indices[0]
        
        features_selected = numpy.take(ant_steps[max_ant[0]][0],numpy.nonzero(ant_steps[max_ant[0]][1]))
        features_selected_count = len(features_selected[0])
        features_selected_score = score_vector[max_ant[0]]
        current_best_solution[0] = features_selected
        current_best_solution[1] = features_selected_score
        current_best_solution[2] = features_selected_count
        #path                     
        current_best_solution[3] = ant_steps[max_ant[0]][0]
        #state
        current_best_solution[4] = ant_steps[max_ant[0]][1]         
        #for i in numpy.take(ants_max_score_indices[0],ants_max_reduction_indices[0]) :
#            features_selected = numpy.take(ant_steps[i][0],numpy.nonzero(ant_steps[i][1]))
#            features_selected_count = len(features_selected[0])
#            features_selected_score = score_vector[i]
#            
#            print(features_selected, features_selected_count, features_selected_score)
#            if(not numpy.array_equal(features_selected,current_best_solution[0]) and 
#               (features_selected_score > current_best_solution[1] or features_selected_count < current_best_solution[2])) :
#                current_best_solution[0] = features_selected
#                current_best_solution[1] = features_selected_score
#                current_best_solution[2] = features_selected_count
#                #path                     
#                current_best_solution[3] = ant_steps[i][0]
#                #state
#                current_best_solution[4] = ant_steps[i][1]
         
        return current_best_solution

    def addPheromone(self, solution) :
    
        for f in range(0,self.FEATURE_COUNT-1):
            current_feature_state  = solution[4][f]
            next_feature_state     = solution[4][f+1]
            current_feature        = solution[3][f]
            next_feature           = solution[3][f+1]

            #if next_feature_state == 1 :
            current_deposit = self.pheromone_matrix[current_feature,next_feature][current_feature_state,next_feature_state]
            error_rate = (1 - solution[1]) * 100
            solution_quality = 1.0 if error_rate == 0 else self.Q / error_rate
            
            deposit = (1 - self.PHI) * current_deposit + solution_quality + (1 - (len(solution[0]) / self.FEATURE_COUNT))
            if deposit <= self.MIN_PHEROMONE : deposit = self.MIN_PHEROMONE
            elif deposit >= self.MAX_PHEROMONE: deposit = self.MAX_PHEROMONE
            
            self.pheromone_matrix[current_feature,next_feature][current_feature_state,next_feature_state] = deposit
              
                            
    def getNextStep(self, prev_steps, prev_steps_states, next_steps):
        current_position = prev_steps[-1]
        current_position_state = prev_steps_states[-1]
        next_steps_scores_0 = dict.fromkeys(next_steps, 0)
        next_steps_scores_1 = dict.fromkeys(next_steps, 0)
        
        for next_step in next_steps:
            next_steps_scores_0[next_step] = self.getNextStepProbability(current_position, current_position_state, next_step, 0, next_steps)
            next_steps_scores_1[next_step] = self.getNextStepProbability(current_position, current_position_state, next_step, 1, next_steps)
        
        next_steps_scores_0_max = max(next_steps_scores_0,key=next_steps_scores_0.get)
        next_steps_scores_1_max = max(next_steps_scores_1,key=next_steps_scores_1.get)
        
        if next_steps_scores_0[next_steps_scores_0_max] == next_steps_scores_1[next_steps_scores_1_max]:
            #step = next_steps_scores_0_max, numpy.random.randint(2)
            step = next_steps_scores_0_max, 0
        elif next_steps_scores_0[next_steps_scores_0_max] > next_steps_scores_1[next_steps_scores_1_max]:
            step = next_steps_scores_0_max, 0
        else:
            step = next_steps_scores_1_max, 1
        
        return step

    def getNextStepProbabilityPheromoneHeuristicWeights(self, current_step, next_step, current_step_state, next_step_state):
        if self.de_running == 1 :
            pheromone_value = self.de_pheromone_matrix[current_step,next_step][current_step_state, next_step_state]
        else:
            pheromone_value = self.pheromone_matrix[current_step,next_step][current_step_state, next_step_state]
        
        #print('computing probability from pheromone and heuristic weights',current_step, current_step_state, next_step, next_step_state)
        #print(pheromone_value,self.heuristic_matrix[current_step,next_step][current_step_state, next_step_state])
  
        return numpy.power(pheromone_value, self.ALPHA) * numpy.power(self.heuristic_matrix[current_step,next_step][current_step_state, next_step_state], self.BETA)

    def getNextStepProbability(self, current_step, current_step_state, next_step, next_step_state, possible_steps):

        n = self.getNextStepProbabilityPheromoneHeuristicWeights(current_step, next_step, current_step_state, next_step_state)
        
        step_0 = 0
        step_1 = 0 

        for step in possible_steps:
            step_0 += self.getNextStepProbabilityPheromoneHeuristicWeights(current_step, step, current_step_state, 0)
            step_1 += self.getNextStepProbabilityPheromoneHeuristicWeights(current_step, step, current_step_state, 1)
        
        d = (step_0 + step_1)
        prob = n / (step_0 + step_1) if (n != 0 and d!= 0) else 0
        
        return prob
    
    def getPheromonePath(self, solution) :
        
        path = []
        
        for f in range(0,self.FEATURE_COUNT-1):
            current_feature_state  = solution[1][f]
            next_feature_state     = solution[1][f+1]
            current_feature        = solution[0][f]
            next_feature           = solution[0][f+1]
            
            deposit = self.pheromone_matrix[current_feature, next_feature][current_feature_state][next_feature_state]
            path.append(deposit)
        
        return path
    
    def updateDEPheromonePath(self, solution, pheromones):
        
        for f in range(0,self.FEATURE_COUNT-1):
            current_feature_state  = solution[1][f]
            next_feature_state     = solution[1][f+1]
            current_feature        = solution[0][f]
            next_feature           = solution[0][f+1]
            self.de_pheromone_matrix[current_feature, next_feature][current_feature_state][next_feature_state] = pheromones[f]
            
    def constructSolutions(self, ant_steps) :
        for step in range(0, ant_steps.shape[0]):
            while True:
                next_steps = numpy.setxor1d(ant_steps[step][0],range(self.FEATURE_COUNT))

                if not len(next_steps) : break
                else :
                   next_step, next_state = self.getNextStep(ant_steps[step][0], ant_steps[step][1], next_steps)
                   ant_steps[step][0] = numpy.append(ant_steps[step][0],[next_step])
                   ant_steps[step][1] = numpy.append(ant_steps[step][1],[next_state])
    
    def runDE(self):
                
        self.de_running = 1
        self.initDEAnts()        
        self.initDEPheromoneMatrix()
        
        for ant in range(0,self.ant_steps.shape[0]):
            
            target_vector = self.getPheromonePath(self.ant_steps[ant])
            
            #random select 3 paths
            paths = random.sample([x for x in range(0, self.ant_steps.shape[0]) if x != ant], 3)
            difference_vector = numpy.subtract(self.getPheromonePath(self.ant_steps[paths[0]]),self.getPheromonePath(self.ant_steps[paths[1]]))
            weighted_difference_vector = numpy.multiply(difference_vector, self.F)
            
            #mutation
            noisy_vector = numpy.add(weighted_difference_vector, paths[2])
            
            #crossover
            trial_vector = []
            
            for i in range(0, len(noisy_vector)) :
                if random.uniform(0.0, 1.0) >= self.CR :
                    trial_vector.append(noisy_vector[i])
                else:
                    trial_vector.append(target_vector[i])
            
            #same paths but varied pheromone levels
            self.updateDEPheromonePath(self.ant_steps[ant], trial_vector)
            
        self.constructSolutions(self.de_ant_steps)
        
        return self.getIterationBestSolution(self.de_ant_steps)
        
    def run(self):
        
        #initialize pheromone matrix, heuristic matrix
        print("Initializing Pheromone Matrix....")
        self.initPheromoneMatrix()
        print("Initializing Heuristic Matrix....")
        self.initHeuristicMatrix()
        
        #while number of iterations not met
        for i in range(0,self.ITERATIONS):

            #randomly place ants
            print("Initializing Ants....")
            self.initAnts()

            #construct solutions
            print("Ants constructing solutions....")
            self.constructSolutions(self.ant_steps)
            
            #evaluate solutions
            print("Getting iteration best solution....")
            local_best = self.getIterationBestSolution(self.ant_steps)
            
            
            #update pheromone
            print("Updating pheromones....")
            
            print("Evaporating pheromones....")
            self.evaporatePheromones()
            
            print("Running DE....")
            de_best_solution = self.runDE()
            self.de_running = 0
            print('iteration best solution', local_best[0], local_best[1], local_best[2])
            print('de best solution', de_best_solution[0], de_best_solution[1], de_best_solution[2])
            
            if de_best_solution[1] > local_best[1]:
                local_best = de_best_solution
            
            self.pheromone_matrix = self.de_pheromone_matrix
            self.addPheromone(local_best)
            
            is_best_solution = 0
            
            if (local_best[1] > self.best_solution[1]) or (not numpy.array_equal(local_best[0], self.best_solution[0]) and  local_best[1] == self.best_solution[1]) :
                self.best_solution = local_best
                is_best_solution = 1
            
            self.evaluateSolution(local_best, is_best_solution)
                
            print('best solution', self.best_solution[0], self.best_solution[1], self.best_solution[2])
            
            print('------------------ Iteration ', i, '------------------')
        
        
        
        self.generateReport()
        
    

accuracy_results = []
precision_results = []
recall_results = []
fr_results = []
individual_run = 20
iteration = 50
for i in range (0,individual_run):
    print('########################### Evaluation ', i, '###########################')
    a = MMASDEFS(iteration, 1.0, 0.5, .049, 6.0, 0.1, 1.0, .9, 0.5)
    a.run()
    print('global best solution ', a.best_solution)
    accuracy_results.append(a.best_result[0])
    precision_results.append(a.best_result[1])
    recall_results.append(a.best_result[2])
    fr_results.append(a.best_result[3])

accuracy_results = numpy.array(accuracy_results, dtype=float)
precision_results = numpy.array(precision_results, dtype=float) 
recall_results = numpy.array(recall_results, dtype=float)  
fr_results = numpy.array(fr_results, dtype=float)  
   
print('average accuracy', numpy.mean(accuracy_results))
print('average precision', numpy.mean(precision_results))
print('average recall', numpy.mean(recall_results))
print('average fr', numpy.mean(fr_results))

