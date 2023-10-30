import 'package:sea_cense/models/cucumber_live.dart';
import 'package:sea_cense/models/cucumber_price.dart';

List<CucumberLive> seaCucumbers = [
  CucumberLive(
      type: 'H.Scabra',
      description:
          'Holothuria scabra, or sandfish, is a species of sea cucumber in the family Holothuriidae. It was placed in the subgenus Metriatyla by Rowe in 1969 and is the CucumberLive (type species of the subgenus). Sandfish are harvested and processed into "beche-de-mer" and eaten in China and other Pacific coastal communities',
      scientificName: 'Holothuria scabra',
      conservationStatus: 'Endangered (Population decreasing) Encyclopedia of Life',
      family: 'Holothuriidae',
      kingdom: 'Animalia',
      phylum: 'Echinoderma',
      diet: 'Diatoms and seaweeds'),
  CucumberLive(
      type: 'H.Spinifera',
      description:
          'Holothuria spinifera, the brown sandfish, is a species of sea cucumber in the family Holothuriidae. It is placed in the subgenus Theelothuria, making its full name Holothuria (Theelothuria) spinifera. In India it is known as cheena attai or raja attai. It lives in tropical regions of the west Indo-Pacific Ocean at depths ranging from 32 to 60 metres (105 to 197 ft). It is fished commercially to produce beche-de-mer.',
      scientificName: 'Holothuria spinifera',
      conservationStatus: '--',
      family: 'Holothuroidea',
      kingdom: 'Animalia',
      phylum: 'Echinodermata',
      diet: '--'),
  CucumberLive(
      type: 'B.Vitiensis',
      description:
          'Bohadschia vitiensis is a species of sea cucumber in the family Holothuriidae. It is also known as the brown sandfish and brown sea cucumber. It is widespread in shallow waters of the Indo-Pacific. It appears to be able to hybridize with Bohadschia argus. Bohadschia vitiensis can grow to 50 cm in total length.',
      scientificName: 'Bohadschia vitiensis',
      conservationStatus: '--',
      family: 'Holothuroidea',
      kingdom: 'Animalia',
      phylum: 'Echinodermata',
      diet: '--'),
  CucumberLive(
      type: 'S.Naso',
      description:
          'It is a classical Stichopus, stout and trapezoidal to rectangular in cross-section and with three rows of podia on the ventral face. Its color is uniform or mottled, from sandy to darj brown, with black lines and dots. Its dorsum wears huge and erected (though retractile) tubercle-like excrescences.Stichopus naso, or also known as tropical holothurian but generally, in modern terms, it is considered a sea cucumber. Stichopus naso was discovered in 1867 specifically in the Philippines. The most recent discovery of this species was in 2011 by the coast of Kagoshima, Kyushu, Japan',
      scientificName: 'Stichopus naso',
      conservationStatus: '--',
      family: 'Holothuroidea',
      kingdom: 'Animalia',
      phylum: 'Echinodermata',
      diet: '--'),
  CucumberLive(
      type: 'Unknown',
      description: '--',
      scientificName: '--',
      conservationStatus: '--',
      family: '--',
      kingdom: '--',
      phylum: '--',
      diet: '--'),
];

Map<String, CucumberPrice> priceCatergories = {
  'img_cat1': CucumberPrice(category: 'img_cat1', price: '600-650', length: '15-20', width: '350-450'),
  'img_cat2': CucumberPrice(category: 'img_cat2', price: '800-900', length: '21-25', width: '451-600'),
  'img_cat3': CucumberPrice(category: 'img_cat3', price: '1100-1300', length: '26-31', width: '601-750'),
  'Unknown': CucumberPrice(category: 'Unknown', price: '--', length: '--', width: '--'),
};
