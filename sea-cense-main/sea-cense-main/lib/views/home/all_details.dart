import 'dart:io';
import 'dart:math';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:sea_cense/models/cucumber_price.dart';
import 'package:sea_cense/viewmodels/cucumber_viewmodel.dart';
import 'package:sea_cense/widgets/divider.dart';
import 'package:sea_cense/widgets/live_detail_row.dart';

class AllDetails extends StatelessWidget {
  const AllDetails({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Consumer<CucumberViewModel>(
        builder: (context, model, child) {
          return Container(
            width: MediaQuery.of(context).size.width,
            height: MediaQuery.of(context).size.height,
            padding: const EdgeInsets.only(top: 40, left: 20, right: 20),
            decoration: const BoxDecoration(
              gradient: LinearGradient(colors: [
                Color(0xFF000613),
                Color(0xFF04223B),
              ], begin: Alignment.topLeft, end: Alignment.centerRight),
            ),
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      GestureDetector(
                        onTap: () {
                          Navigator.pop(context);
                        },
                        child: const Padding(
                          padding: EdgeInsets.only(top: 40),
                          child: Icon(
                            Icons.arrow_back_ios,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      GestureDetector(
                        onTap: () {
                          model.getSharableScreen(allDetailsReport(context, model), context);
                        },
                        child: const Padding(
                          padding: EdgeInsets.only(top: 40),
                          child: Icon(
                            Icons.share,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                  allDetailsReport(context, model),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  allDetailsReport(BuildContext context, CucumberViewModel model) {
    return Container(
      padding: const EdgeInsets.only(left: 20, right: 20),
      margin: const EdgeInsets.only(top: 20),
      decoration: const BoxDecoration(
        borderRadius: BorderRadius.all(Radius.circular(20)),
        gradient: LinearGradient(colors: [
          Color(0xFF0D1C3A),
          Color(0xFF172531),
        ], begin: Alignment.topLeft, end: Alignment.centerRight),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            height: MediaQuery.of(context).size.height / 4,
            margin: const EdgeInsets.only(top: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: const BorderRadius.all(
                Radius.circular(20),
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.3),
                  spreadRadius: 5,
                  blurRadius: 7,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20.0),
              child: Image.file(File(model.imageFile!.path)),
            ),
          ),
          model.cucumberAll!.cucumberLive! == "Unknown"
              ? Container()
              : Container(
                  width: MediaQuery.of(context).size.width,
                  margin: const EdgeInsets.only(top: 20),
                  padding: const EdgeInsets.all(10),
                  decoration: const BoxDecoration(
                    color: Color(0x29000000),
                    borderRadius: BorderRadius.all(Radius.circular(20)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        model.cucumberAll!.cucumberLive.type,
                        style: const TextStyle(
                            color: Color(0xD20099FF),
                            fontSize: 30,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 3.0),
                      ),
                      const SizedBox(
                        height: 5,
                      ),
                      Text(
                        model.cucumberAll!.cucumberLive.scientificName,
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            letterSpacing: 3.0,
                            fontStyle: FontStyle.italic),
                      ),
                      const CustomDivider(),
                      Text(
                        model.cucumberAll!.cucumberLive.description,
                        style: const TextStyle(color: Colors.white, fontSize: 14, letterSpacing: 3.0),
                      ),
                      LiveDetailRow(
                        title: 'Conservation Status',
                        desc: model.cucumberAll!.cucumberLive.conservationStatus!,
                      ),
                      LiveDetailRow(
                        title: 'Family',
                        desc: model.cucumberAll!.cucumberLive.family,
                      ),
                      LiveDetailRow(
                        title: 'Kingdom',
                        desc: model.cucumberAll!.cucumberLive.kingdom,
                      ),
                      LiveDetailRow(
                        title: 'Phylum',
                        desc: model.cucumberAll!.cucumberLive.phylum,
                      ),
                      LiveDetailRow(
                        title: 'Diet',
                        desc: model.cucumberAll!.cucumberLive.diet,
                      ),
                      const SizedBox(
                        height: 10,
                      ),
                    ],
                  ),
                ),
          model.cucumberAll!.cucumberPrice! == "Unknown"
              ? Container()
              : Container(
                  width: MediaQuery.of(context).size.width,
                  margin: const EdgeInsets.only(top: 20),
                  padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
                  decoration: const BoxDecoration(
                    color: Color(0x29000000),
                    borderRadius: BorderRadius.all(Radius.circular(20)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      approxValRow(
                          Icons.price_change_outlined,
                          'Price',
                          '${getRandomDivisibleByFive(model.cucumberAll!.cucumberPrice!)} LKR',
                          context,
                          false,
                          special: true),
                      approxValRow(
                          Icons.crop_16_9_rounded,
                          'Length',
                          '${generateRandomNumber(model.cucumberAll!.cucumberPrice!.category!, model.cucumberAll!.cucumberPrice!.length!)} cm',
                          context,
                          false),
                      approxValRow(
                          Icons.crop_16_9_rounded,
                          'Width',
                          '${generateRandomNumber(model.cucumberAll!.cucumberPrice!.category!, model.cucumberAll!.cucumberPrice!.width!)} g',
                          context,
                          true),
                      const SizedBox(
                        height: 5,
                      ),
                      const CustomDivider(),
                      LiveDetailRow(
                        title: 'Price range (LKR)',
                        desc: model.cucumberAll!.cucumberPrice!.price!,
                      ),
                      LiveDetailRow(
                        title: 'Length range (cm)',
                        desc: model.cucumberAll!.cucumberPrice!.length!,
                      ),
                      LiveDetailRow(
                        title: 'Width range (g)',
                        desc: model.cucumberAll!.cucumberPrice!.width!,
                      ),
                      const SizedBox(
                        height: 20,
                      ),
                    ],
                  ),
                ),
          model.cucumberAll!.cucumberJuvenile! == "Adult" || model.cucumberAll!.cucumberJuvenile! == "Unknown"
              ? Container()
              : Container(
                  width: MediaQuery.of(context).size.width,
                  margin: const EdgeInsets.only(top: 20),
                  padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 10),
                  decoration: const BoxDecoration(
                    color: Color(0x29000000),
                    borderRadius: BorderRadius.all(Radius.circular(20)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        model.cucumberAll!.cucumberJuvenile!.initSize!,
                        style: const TextStyle(
                            color: Color(0xD20099FF),
                            fontSize: 30,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 3.0),
                      ),
                      const SizedBox(
                        height: 10,
                      ),
                      const CustomDivider(),
                      LiveDetailRow(
                        title: 'Final Weight (g)',
                        desc: model.cucumberAll!.cucumberJuvenile!.weight!,
                      ),
                      LiveDetailRow(
                        title: 'Growth rate per day (%)',
                        desc: model.cucumberAll!.cucumberJuvenile!.growthRate!,
                      ),
                      LiveDetailRow(
                        title: 'Survival rate (%)',
                        desc: model.cucumberAll!.cucumberJuvenile!.survivalRate!,
                      ),
                      LiveDetailRow(
                        title: 'Total Biomass (g)',
                        desc: model.cucumberAll!.cucumberJuvenile!.totalBiomass!,
                      ),
                    ],
                  ),
                ),
          const SizedBox(
            height: 20,
          ),
        ],
      ),
    );
  }

  approxValRow(IconData icon, String title, String value, BuildContext context, bool transform,
      {bool special = false}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      margin: const EdgeInsets.symmetric(vertical: 15),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          transform
              ? Transform.rotate(
                  angle: 90 * math.pi / 180,
                  child: Icon(
                    icon,
                    size: 20,
                    color: Colors.white,
                  ),
                )
              : Icon(
                  icon,
                  size: 20,
                  color: Colors.white,
                ),
          SizedBox(
            width: MediaQuery.of(context).size.width / 4,
            child: Text(
              title,
              style: const TextStyle(
                color: Colors.white54,
                fontSize: 20,
                fontWeight: FontWeight.w300,
                letterSpacing: 3.0,
              ),
            ),
          ),
          Container(
            height: 20,
            width: 0.5,
            color: Colors.white,
          ),
          SizedBox(
            width: MediaQuery.of(context).size.width / 4,
            child: Text(
              value,
              style: TextStyle(
                  color: special ? const Color(0xD218C5A8) : const Color(0xD20099FF),
                  fontSize: 20,
                  fontWeight: FontWeight.w500,
                  letterSpacing: 3.0),
            ),
          ),
        ],
      ),
    );
  }

  dynamic getRandomDivisibleByFive(CucumberPrice model) {
    if (model.category == 'Unkown') {
      return '--';
    }
    List<String> priceParts = model.price!.split('-');
    final random = Random();
    int result, min, max;
    min = int.parse(priceParts[0]);
    max = int.parse(priceParts[1]);

    do {
      result = min + random.nextInt(max - min + 1);

      result -= result % 5;
    } while (result < min);

    return result;
  }

  dynamic generateRandomNumber(String category, String value) {
    if (category == 'Unkown') {
      return '--';
    }
    List<String> parts = value.split('-');
    final random = Random();
    int min = int.parse(parts[0]);
    int max = int.parse(parts[1]);
    return min + random.nextInt(max - min + 1);
  }
}
