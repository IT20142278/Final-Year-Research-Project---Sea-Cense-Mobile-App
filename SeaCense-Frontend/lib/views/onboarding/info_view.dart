import 'package:flutter/material.dart';
import 'package:sea_cense/views/onboarding/onboarding.dart';
import 'package:sea_cense/widgets/common_button_widget.dart';

class InfoView extends StatelessWidget {
  const InfoView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SizedBox(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              'About us',
              style: TextStyle(
                fontSize: 42,
                fontWeight: FontWeight.w200,
              ),
            ),
            Image.asset(
              'assets/images/info_banner.png',
              scale: 4,
            ),
            Container(
              width: MediaQuery.of(context).size.width / 1.3,
              padding: const EdgeInsets.only(top: 80, right: 30),
              child: const Text(
                'Sea Cense is a mobile application solution aimed at improving the identification and processing of live and dried sea cucumbers while offering an effective prediction system to determine the survival and growth rate of juvenile sea cucumbers',
                textAlign: TextAlign.justify,
                style: TextStyle(
                  fontSize: 18,
                  color: Color(0xFF2743FD),
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
            Container(
              width: MediaQuery.of(context).size.width,
              margin: const EdgeInsets.only(right: 40, top: 40),
              alignment: Alignment.centerRight,
              child: CommonButtonWidget(
                text: "Next",
                size: MediaQuery.of(context).size.width / 3,
                onTap: () {
                  Navigator.of(context).push(MaterialPageRoute(builder: (context) => const OnboardingView()));
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
