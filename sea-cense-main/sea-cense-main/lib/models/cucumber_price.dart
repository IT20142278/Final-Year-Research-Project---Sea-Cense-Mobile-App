class CucumberPrice {
  String? width;
  String? price;
  String? length;
  String? category;

  CucumberPrice({
    this.price,
    this.width,
    this.length,
    this.category,
  });

  CucumberPrice.fromJson(Map<String, dynamic> json) {
    width = json['width'];
    price = json['price'];
    length = json['length'];
    category = json['category'];
  }
  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = <String, dynamic>{};
    data['width'] = width;
    data['price'] = price;
    data['length'] = length;
    data['category'] = category;

    return data;
  }
}
