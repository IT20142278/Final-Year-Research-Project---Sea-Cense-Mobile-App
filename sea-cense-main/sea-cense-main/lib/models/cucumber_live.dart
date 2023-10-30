class CucumberLive {
  String? type;
  String? diet;
  String? family;
  String? phylum;
  String? kingdom;
  String? description;
  String? scientificName;
  String? conservationStatus;

  CucumberLive({
    this.type,
    this.diet,
    this.family,
    this.phylum,
    this.kingdom,
    this.description,
    this.scientificName,
    this.conservationStatus,
  });

  CucumberLive.fromJson(Map<String, dynamic> json) {
    type = json['type'];
    diet = json['diet'];
    family = json['family'];
    phylum = json['phylum'];
    kingdom = json['kingdom'];
    description = json['description'];
    scientificName = json['scientificName'];
    conservationStatus = json['conservationStatus'];
  }
  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = <String, dynamic>{};
    data['type'] = type;
    data['diet'] = diet;
    data['family'] = family;
    data['phylum'] = phylum;
    data['kingdom'] = kingdom;
    data['description'] = description;
    data['scientificName'] = scientificName;
    data['conservationStatus'] = conservationStatus;

    return data;
  }
}
