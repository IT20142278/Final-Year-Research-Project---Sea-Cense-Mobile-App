import 'dart:io';
import 'dart:math';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:sea_cense/models/cucumber_price.dart';
import 'package:sea_cense/viewmodels/cucumber_viewmodel.dart';
import 'package:sea_cense/widgets/divider.dart';
import 'package:sea_cense/widgets/live_detail_row.dart';

class PriceDetails extends StatelessWidget {
  const PriceDetails({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        padding: const EdgeInsets.only(top: 40, left: 20, right: 20),
        decoration: const BoxDecoration(
          gradient: LinearGradient(colors: [
            Color.fromARGB(255, 24, 116, 192),
            Color.fromARGB(255, 60, 77, 108),
          ], begin: Alignment.topLeft, end: Alignment.centerRight),
        ),
        child: Consumer<CucumberViewModel>(
          builder: (context, model, child) {
            return SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Row(
                    children: [
                      GestureDetector(
                        onTap: () {
                          Navigator.pop(context);
                        },
                        child: Container(
                          alignment: Alignment.centerLeft,
                          margin: const EdgeInsets.only(top: 40),
                          child: const Icon(
                            Icons.arrow_back_ios,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      Container(
                        width: MediaQuery.of(context).size.width - 70,
                        alignment: Alignment.center,
                        margin: const EdgeInsets.only(top: 40),
                        child: const Text(
                          "Price Prediction",
                          style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
                        ),
                      )
                    ],
                  ),
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
                  Container(
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
                        approxValRow(Icons.price_change_outlined, 'Price',
                            '${getRandomDivisibleByFive(model.cucumberPrice!)} LKR', context, false,
                            special: true),
                        approxValRow(
                            Icons.crop_16_9_rounded,
                            'Length',
                            '${generateRandomNumber(model.cucumberPrice!.category!, model.cucumberPrice!.length!)} cm',
                            context,
                            false),
                        approxValRow(
                            Icons.crop_16_9_rounded,
                            'Width',
                            '${generateRandomNumber(model.cucumberPrice!.category!, model.cucumberPrice!.width!)} g',
                            context,
                            true),
                        const SizedBox(
                          height: 5,
                        ),
                        const CustomDivider(),
                        LiveDetailRow(
                          title: 'Price range (LKR)',
                          desc: model.cucumberPrice!.price!,
                        ),
                        LiveDetailRow(
                          title: 'Length range (cm)',
                          desc: model.cucumberPrice!.length!,
                        ),
                        LiveDetailRow(
                          title: 'Width range (g)',
                          desc: model.cucumberPrice!.width!,
                        ),
                        const SizedBox(
                          height: 20,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        ),
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
                color: Colors.white,
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
                  color: special ? const Color(0xD218CAAC) : const Color(0xD216A1FD),
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
