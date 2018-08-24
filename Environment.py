import numpy as np 

def draw_card():
    color, number = None, None
    if np.random.randint(0,2) == 1:
        color = "Black"
    else: 
        color = "Red"
    return {"color" : color, "number": np.random.randint(1,11)}


class Easy21:
    
    def __init__(self):
        self.minValue, self.maxValue = 1,11
        self.dealerUpperbound, self.gameUpperbound, self.gameLowerbound= 17, 21, 1
        self.state = {"dealer" : np.random.randint(self.minValue,self.maxValue), "player_sum" : np.random.randint(self.minValue,self.maxValue)}
        self.isTerminal = False
    
    def isTerminal(self):
        "Check if the game is finished"
        
        return self.isTerminal
            
    def step(self, action):
        
        if action == "Hit":
            player_sum, dealer = self.state["player_sum"], self.state["dealer"]
            card = draw_card()
            if card["color"] == "Black":
                player_sum+=card["number"]
            else:
                player_sum-=card["number"]
            self.state = {"dealer": dealer, "player_sum": player_sum}
            if player_sum>self.gameUpperbound or player_sum<self.gameLowerbound:
                self.isTerminal = True
                return self.state, -1
            else:
                return self.state, 0
            #self.print_state()
                
        elif action == "Stick":
            """Dealer's turn to play"""
            while 1==1:
                player_sum, dealer = self.state["player_sum"], self.state["dealer"]
                card = draw_card()
                if card["color"] == "Black":
                    dealer+=card["number"]
                else:
                    dealer-=card["number"]
                self.state = {"dealer": dealer, "player_sum": player_sum}

                if dealer>self.gameUpperbound or dealer < self.gameLowerbound:
                    self.isTerminal = True

                    return self.state, +1
                    
                    
                if dealer>=self.dealerUpperbound:
                    """The game stops, let's compare our numbers"""
                    if dealer>player_sum:
                        self.isTerminal = True

                        return self.state, -1

                    elif dealer == player_sum:
                        self.isTerminal = True
                        return self.state, 0
                    else:
                        self.isTerminal = True

                        return self.state, +1

    def print_state(self):
        print("Dealer's number: {}, Your number: {}".format(self.state["dealer"], self.state["player_sum"]))