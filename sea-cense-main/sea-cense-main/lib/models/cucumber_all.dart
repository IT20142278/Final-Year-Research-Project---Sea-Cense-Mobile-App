// ignore_for_file: public_member_api_docs, sort_constructors_first
import 'package:sea_cense/models/cucumber_juvenile.dart';
import 'package:sea_cense/models/cucumber_live.dart';
import 'package:sea_cense/models/cucumber_price.dart';

class CucumberAll {
  dynamic cucumberLive;
  dynamic cucumberJuvenile;
  dynamic cucumberPrice;

  CucumberAll({
    this.cucumberLive,
    this.cucumberJuvenile,
    this.cucumberPrice,
  });

  CucumberAll.fromJson(Map<String, dynamic> json) {
    cucumberLive = json['live-classifier'];
    cucumberJuvenile = json['age'];
    cucumberPrice = json['price'];
  }
  Map<String, dynamic> toJson() {
    final Map<String, dynamic> data = <String, dynamic>{};
    data['live-classifier'] = cucumberLive;
    data['age'] = cucumberJuvenile;
    data['price'] = cucumberPrice;

    return data;
  }

  CucumberAll copyWith({
    CucumberLive? cucumberLive,
    CucumberJuvenile? cucumberJuvenile,
    CucumberPrice? cucumberPrice,
  }) {
    return CucumberAll(
      cucumberLive: cucumberLive ?? this.cucumberLive,
      cucumberJuvenile: cucumberJuvenile ?? this.cucumberJuvenile,
      cucumberPrice: cucumberPrice ?? this.cucumberPrice,
    );
  }
}
