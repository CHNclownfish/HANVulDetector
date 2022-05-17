class showRes:
    def __init__(self, losses, scores):
        self.losses = losses
        self.scores = scores

    def showLoss(self):
        for t in self.losses:
            print('the following is losses for '+ t)
            for i, x in enumerate(self.losses[t]):
                print(x)

    def showScores(self):
        for t in self.scores:
            print('the following is scores for '+ t)
            for i, x in enumerate(self.scores[t]):
                print(x)