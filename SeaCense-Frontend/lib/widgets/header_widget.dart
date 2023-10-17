import 'package:flutter/material.dart';

class HeaderWidget extends StatelessWidget {
  final String text;
  final bool showBack;
  const HeaderWidget({super.key, required this.text, this.showBack = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: MediaQuery.of(context).size.width,
      height: MediaQuery.of(context).size.height / 3,
      alignment: Alignment.centerLeft,
      child: Stack(
        children: [
          Container(
            padding: const EdgeInsets.only(left: 10, top: 40),
            width: MediaQuery.of(context).size.width / 1.7,
            decoration: const BoxDecoration(
              image: DecorationImage(
                image: AssetImage("assets/images/onBoarding_header_bg.png"),
                fit: BoxFit.fill,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const SizedBox(
                      width: 10,
                    ),
                    showBack
                        ? GestureDetector(
                            onTap: () {
                              Navigator.pop(context);
                            },
                            child: const Icon(
                              Icons.arrow_back_ios,
                              color: Colors.white,
                            ),
                          )
                        : Container(
                            width: 20,
                          ),
                    const SizedBox(
                      width: 10,
                    ),
                    Image.asset(
                      'assets/images/logo_without_text.png',
                      height: 90,
                    ),
                  ],
                ),
                Text(
                  text,
                  style: const TextStyle(color: Colors.white, fontSize: 30, fontWeight: FontWeight.w200),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}
