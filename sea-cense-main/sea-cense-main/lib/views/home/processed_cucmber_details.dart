import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:sea_cense/viewmodels/cucumber_viewmodel.dart';
import 'package:sea_cense/widgets/divider.dart';
import 'package:sea_cense/widgets/live_detail_row.dart';

class ProcessedCucumberDetails extends StatelessWidget {
  const ProcessedCucumberDetails({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        padding: const EdgeInsets.only(top: 40, left: 20, right: 20),
        decoration: const BoxDecoration(
          gradient: LinearGradient(colors: [
            Color(0xFF1869C0),
            Color(0xFF443C6C),
          ], begin: Alignment.topLeft, end: Alignment.centerRight),
        ),
        child: Consumer<CucumberViewModel>(
          builder: (context, model, child) {
            return SingleChildScrollView(
              child: Column(
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
                          "Quality Prediction",
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
                    padding: const EdgeInsets.all(10),
                    decoration: const BoxDecoration(
                      color: Color(0x29000000),
                      borderRadius: BorderRadius.all(Radius.circular(20)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          model.cucumberProcessed!.predictedType!,
                          style: const TextStyle(
                              color: Color(0xD216A1FD),
                              fontSize: 30,
                              fontWeight: FontWeight.bold,
                              letterSpacing: 3.0),
                        ),
                        const CustomDivider(),
                        LiveDetailRow(
                          title: 'Specimen quality',
                          desc: model.cucumberProcessed!.predictedClass!,
                        ),
                        const SizedBox(
                          height: 10,
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
}
