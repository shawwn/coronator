import os
import numpy as np
import threading
from imageio import imwrite,imread
from scipy.ndimage import zoom

class Corona():
    def __init__(self):
        self.infectionProbability = 0.02
        self.lethalProbability = 0.02
        
        self.stepsUntilContagious_min = 10
        self.stepsUntilContagious_max = 30
        
        self.stepsUntilSymptoms_min = 50
        self.stepsUntilSymptoms_max = 80
        
        self.stepsUntilHealed_min = 200
        self.stepsUntilHealed_max = 300
        
        self.stepsUntilReceptive_min = 500
        self.stepsUntilReceptive_max = 700
        
        self.contaminationRadius = 1
        
        self.symptoms_show = 0.5
        
        self.travel_distance = 1      
        self.travel_probability = 0.2  

        self.air_travel_probability = 0.0
        self.social_distance_probability = 0.0
        
        #self.no_symptoms_avoid_others = 0.1
        #self.symptoms_avoid_others = 0.75
        #self.immune_avoid_others = 0.01
          
class World():
    def __init__(self,corona=None,world_width=1024,world_height=1024,agents=1000,initialInfections=10):
        self.states = ["HEALTHY_NOT_IMMUNE","INFECTED_NO_SYMPTOMS","INFECTED_SYMPTOMS","HEALTHY_IMMUNE","DEAD"]
        self.colors = [[86,180,233],[240,228,66],[213,94,0],[0,114,178],[204,121,167]]
        
        self.emojis = []
        self.emojis.append(imread("emojis/nonimmune.png",pilmode="RGB"))
        self.emojis.append(imread("emojis/contagious.png",pilmode="RGB"))
        self.emojis.append(imread("emojis/symptoms.png",pilmode="RGB"))
        self.emojis.append(imread("emojis/immune.png",pilmode="RGB"))
        self.emojis.append(imread("emojis/dead.png",pilmode="RGB"))
        
        self.emojiScale = self.emojis[0].shape[0]
        
        self.world = np.zeros((world_height,world_width),dtype=int)-1
        self.corona = corona
        
        self.agents = []
        startingPositions = np.random.choice(np.arange(world_width*world_height),size=16,replace=False)
        infected = np.random.choice(np.arange(agents),size=initialInfections,replace=False)
        for i in range(16):
            a = Agent(startingPositions[i]%world_width,int(startingPositions[i]/world_width),1 if i in infected else 0,will_show_symptoms=np.random.random()<corona.symptoms_show,will_die_if_infected=np.random.random()<corona.lethalProbability)  
            self.world[a.y,a.x] = len(self.agents)
            self.agents.append(a)
            
        for i in range(16,agents):
            found = False
            while not found:
                a = self.agents[np.random.randint(i)]    
                x = a.x + np.random.randint(-2,3)
                y = a.y + np.random.randint(-2,3)
                if x>=0 and x < self.world.shape[1] and y>=0 and y < self.world.shape[0] and self.world[y,x]==-1:
                    found =True
                    a = Agent(x,y,1 if i in infected else 0,will_show_symptoms=np.random.random()<corona.symptoms_show,will_die_if_infected=np.random.random()<corona.lethalProbability)  
                    self.world[a.y,a.x] = len(self.agents)
                    self.agents.append(a)
                    
    def update(self):
        order = np.random.permutation(len(self.agents))
        for i in order:
            
            self.agents[i].update(self.world,self.corona,self.agents)    

    def render(self):
        t = np.zeros((self.world.shape[0]*self.emojiScale,self.world.shape[1]*self.emojiScale,3),dtype=np.uint8)+255
        for agent in self.agents:
            t[agent.y*self.emojiScale:agent.y*self.emojiScale+self.emojiScale,agent.x*self.emojiScale:agent.x*self.emojiScale+self.emojiScale] = self.emojis[agent.state]             
        return t
        
class Agent():
    def __init__(self,x,y,state,will_show_symptoms,will_die_if_infected):
        self.state = state
        self.infection_time = -1 if state==0 else 0
        self.x = x
        self.y = y
        self.will_show_symptoms = will_show_symptoms
        self.will_die = will_die_if_infected
        self.target_x = x
        self.target_y = y
        
        
        self.is_contagious = False
        
    def update(self,world,corona,agents):
        if self.state==4:
            return
            
        if self.infection_time>-1:
            self.infection_time+=1
        
        
        if (self.x == self.target_x and self.y == self.target_y) or np.random.random()<0.2:
                x_from = max(0,self.x-corona.travel_distance)
                x_to = min(world.shape[1],self.x+corona.travel_distance+1)
                y_from = max(0,self.y-corona.travel_distance)
                y_to = min(world.shape[0],self.y+corona.travel_distance+1)   
            
                self.target_x = np.random.randint(x_from,x_to) 
                self.target_y = np.random.randint(y_from,y_to)   
            
        if self.state == 1:
            if self.will_show_symptoms:
                th = np.random.randint(corona.stepsUntilSymptoms_min,corona.stepsUntilSymptoms_max)
                if self.infection_time > th:
                    self.state = 2
            th = np.random.randint(corona.stepsUntilHealed_min,corona.stepsUntilHealed_max)            
            if self.infection_time > th:
                self.state = 3
                self.is_contagious = False
                
        elif self.state==2:
            if self.will_die and np.random.random()<0.1:
                self.state=4
                self.infection_time = -1
                self.is_contagious = False
                return
                
            else:
                th = np.random.randint(corona.stepsUntilHealed_min,corona.stepsUntilHealed_max)            
                if self.infection_time > th:
                    self.is_contagious = False
                    if self.will_die:
                        self.infection_time = -1
                        self.state=4
                        return
                    else:
                        self.state = 3
        
        elif self.state==3:
            th = np.random.randint(corona.stepsUntilReceptive_min,corona.stepsUntilReceptive_max)            
            if self.infection_time > th:
                self.state = 0                
                    
        if not self.is_contagious and (self.state==1 or self.state==2):

            th = np.random.randint(corona.stepsUntilContagious_min,corona.stepsUntilContagious_max)        
            #print("check contagious",self.infection_time,th)    
            if self.infection_time > th:
                self.is_contagious = True
                
        x_from = max(0,self.x-corona.contaminationRadius)
        x_to = min(world.shape[1],self.x+corona.contaminationRadius+1)
        y_from = max(0,self.y-corona.contaminationRadius)
        y_to = min(world.shape[0],self.y+corona.contaminationRadius+1)
        
        if self.state == 0:
            for y in range(y_from,y_to):    
                if self.state > 0:
                    break                     
                for x in range(x_from,x_to):  
                    if x==self.x and y==self.y:
                        continue
                    if world[y,x]>-1 and agents[world[y,x]].is_contagious:
                        #print("check infection")
                        if np.random.random()<corona.infectionProbability:
                            #print("infected!")
                            self.state = 1
                            self.infection_time = 0  
                            break
        
        if self.state != 2 and np.random.random() < corona.travel_probability:
            if np.random.random()<corona.air_travel_probability:
                empty = np.where(world==-1)
                idx = np.random.randint(len(empty[0]))
                
                px = empty[1][idx]
                py = empty[0][idx]
                world[py,px] = world[self.y,self.x]
                world[self.y,self.x] = -1
                self.x = px
                self.y = py
                    
            else:
                x_from = max(0,self.x-1)
                x_to = min(world.shape[1],self.x+2)
                y_from = max(0,self.y-1)
                y_to = min(world.shape[0],self.y+2)                    
                bestOption=None
                
                if np.random.random()<corona.social_distance_probability:
                    bestScore = -1
                    for y in range(y_from,y_to):    
                        for x in range(x_from,x_to):                       
                            
                            
                            if world[y,x]==-1 or (x==self.x and y==self.y):
                                score = np.sum(world[y-1:y+2,x-1:x+2]==-1)
                                if score>bestScore:
                                    bestScore = score
                                    bestOption = [[x,y]]
                                elif score==bestScore:
                                    bestOption.append([x,y])
                              
                                
                
                else:  
                    bestScore = 1000000000
                    for y in range(y_from,y_to):    
                        for x in range(x_from,x_to):               
                            if world[y,x]==-1 or (x==self.x and y==self.y):
                                dx = self.target_x - x
                                dy = self.target_y - y
                                if dx*dx+dy*dy<bestScore:
                                    bestScore = dx*dx+dy*dy
                                    bestOption = [[x,y]]
                                elif dx*dx+dy*dy==bestScore:
                                    bestOption.append([x,y])   
                                 
                if len(bestOption)==1:
                    bo = bestOption[0]
                else:
                    bo = bestOption[np.random.randint(len(bestOption))]      
                                                  
                if bo[1]!=self.y or bo[0]!=self.x:
                    world[bo[1],bo[0]] = world[self.y,self.x]
                    world[self.y,self.x] = -1
                    self.x = bo[0]
                    self.y = bo[1]
                                




def run(rounds=3000,world_width = 32,world_height = 32):
    exportFolder = "corona_export_01/"
    if not os.path.exists(exportFolder):
        os.mkdir(exportFolder)
    fileIndex = 0
    corona = Corona()
    world = World(corona=corona,world_width=world_width,world_height=world_height,agents=700,initialInfections=1)
    for i in range(rounds):
        print(i,"/",rounds)
        world.update()
        tex = world.render()
        imwrite(exportFolder+str(fileIndex).zfill(5)+".jpg",zoom(tex,(0.75,0.75,1),order=0))
        fileIndex += 1
  
run()
