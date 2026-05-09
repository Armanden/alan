class SimplePatternAI:
    def predict_next(self, numbers):
        if len(numbers) < 2:
            return None

        # Find differences
        diffs = []
        for i in range(len(numbers)-1):
            diffs.append(numbers[i+1] - numbers[i])

        # Check if pattern is constant
        if all(d == diffs[0] for d in diffs):
            return numbers[-1] + diffs[0]

        return "Pattern unclear"


ai = SimplePatternAI()

print(ai.predict_next([1,2,3,5]))
