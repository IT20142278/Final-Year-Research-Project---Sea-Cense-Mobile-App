class CucumberJuvenile {
  String? type;
  String? weight;
  String? initSize;
  String? growthRate;
  String? survivalRate;
  String? totalBiomass;

  CucumberJuvenile({
    this.type,
    this.weight,
    this.initSize,
    this.growthRate,
    this.survivalRate,
    this.totalBiomass,
  });

  CucumberJuvenile.fromJson(Map<String, dynamic> json) {
    type = json['class'];
    weight = json['finalWeight'];
    initSize = json['initSizeGroup'];
    survivalRate = json['survivalRate'];
    totalBiomass = json['totalBioMass'];
    growthRate = json['growthRatePerDay'];
  }
  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = <String, dynamic>{};
    data['class'] = type;
    data['finalWeight'] = weight;
    data['initSizeGroup'] = initSize;
    data['survivalRate'] = survivalRate;
    data['totalBioMass'] = totalBiomass;
    data['growthRatePerDay'] = growthRate;

    return data;
  }
}
