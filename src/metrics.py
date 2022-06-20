class MetricHandler:
	def __init__(self):
		self.no_predictions = 0
		self.correct_predictions = 0
		self.no_tags_predicted = 0
		self.tag_occurence = {}
		
	def new_prediction(self, tags):
		self.no_predictions += 1
		for tag in tags:
			self.no_tags_predicted += 1
			if tag in self.tag_occurence:
				self.tag_occurence[tag] += 1
			else:
				self.tag_occurence[tag] = 1
				
		
	def feedback(self, correct):
		if correct:
			self.correct_predictions += 1
		
	def get_no_predictions(self):
		return self.no_predictions
		
	def get_no_correct_predictions(self):
		return self.correct_predictions
		
	def get_tag_occurences(self):
		print(self.tag_occurence)
		return self.tag_occurence, len(self.tag_occurence), self.no_tags_predicted
