import heapq

class Scheduler:
    def __init__(self):
        self.heap=[]
        self.counter=0
    
    def push(self, timestamp:float , agent_id:int , action: str):
        heapq.heappush(self.heap, (timestamp, self.counter, agent_id, action))
        self.counter += 1
#Counter is addded for safty measuere as when say both  tiemstamp and agent_id become same and action is complex then for heap to compare we add the counter
#when heap compaares with agent id it will pop the min first irrespective of what was added fisrt hence counter is added

    def pop(self):
        if not self.heap:
            return None
        timestamp, _, agent_id, action = heapq.heappop(self.heap)#_ means yes give the value but i dodnt need it so discard it
        return timestamp, agent_id, action
        