class CucumberProcessed {
  String? predictedType;
  String? predictedClass;
  String? predictedProbabilities;

  CucumberProcessed({
    this.predictedType,
    this.predictedClass,
    this.predictedProbabilities,
  });

  CucumberProcessed.fromJson(Map<String, dynamic> json) {
    predictedClass = json['quality'];
    predictedType = json['predicted_name'];
    predictedProbabilities = json['predicted_probabilities'];
  }
  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = <String, dynamic>{};
    data['quality'] = predictedClass;
    data['predicted_name'] = predictedType;
    data['predicted_probabilities'] = predictedProbabilities;

    return data;
  }
}
